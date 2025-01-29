import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import sqlite3
import logging
from web3 import Web3
from tqdm import tqdm
import os
from dotenv import load_dotenv
import time
import argparse
import json
from typing import List

# Load environment variables
load_dotenv()

# Configuration
RPC_ENDPOINT = os.getenv('RPC_ENDPOINT')
CHAIN_ID = int(os.getenv('CHAIN_ID', '2741'))
DB_PATH = os.getenv('DB_PATH', 'distribution.db')
DB_LOG_PATH = 'minting.log'

# NFT Contract Configuration
CONTRACT_ADDRESS = Web3.to_checksum_address('0xe501994195b9951413411395ed1921a88eff694e')
MINT_VALUE = 0.00033  # ETH value being sent ($1.02)
GAS_LIMIT = 511351  # Gas limit
BASE_FEE = Web3.to_wei(0.04525, 'gwei')
MAX_PRIORITY_FEE = Web3.to_wei(0.04525, 'gwei')
MAX_FEE = Web3.to_wei(0.04525, 'gwei')

class MerkleTree:
    """Merkle tree implementation compatible with Solady's library"""
    def __init__(self, leaves: List[str]):
        self.leaves = [Web3.to_bytes(hexstr=leaf) for leaf in sorted(leaves)]
        self.tree = []
        self.build_tree()

    def build_tree(self):
        current_level = self.leaves.copy()
        self.tree.append(current_level)
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i+1] if (i+1) < len(current_level) else left
                # Sort siblings before hashing
                if left > right:
                    left, right = right, left
                combined = left + right
                next_node = Web3.keccak(combined)
                next_level.append(next_node)
            current_level = next_level
            self.tree.append(current_level)

    def get_proof(self, index: int) -> List[str]:
        proof = []
        current_index = sorted(self.leaves).index(self.leaves[index])
        
        for level in self.tree[:-1]:
            level_len = len(level)
            if current_index % 2 == 1:
                sibling_index = current_index - 1
            else:
                sibling_index = current_index + 1 if current_index + 1 < level_len else current_index
            
            proof.append(Web3.to_hex(level[sibling_index]))
            current_index = current_index // 2
            
        return proof

    @property
    def root(self) -> str:
        return Web3.to_hex(self.tree[-1][0]) if self.tree else ""

class NFTMinter:
    def __init__(self):
        self.web3 = Web3(Web3.HTTPProvider(RPC_ENDPOINT))
        self.contract = self.setup_contract()
        self.merkle_root = None
        self.setup_logging()
        self.should_stop = False
        self.setup_database()
        self.setup_merkle_data()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(DB_LOG_PATH), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        """Setup database tables if they don't exist"""
        with sqlite3.connect(DB_PATH) as conn:
            # Minting status table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS minting_status (
                    wallet_address TEXT PRIMARY KEY,
                    status TEXT CHECK(status IN ('pending', 'success', 'failed')) DEFAULT 'pending',
                    tx_hash TEXT,
                    error_message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Merkle proofs table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS merkle_proofs (
                    wallet_address TEXT PRIMARY KEY,
                    proof TEXT,
                    root_hash TEXT
                )
            ''')
            
            # Initialize minting status
            conn.execute('''
                INSERT OR IGNORE INTO minting_status (wallet_address, status)
                SELECT address, 'pending'
                FROM wallets
                WHERE wallet_type = 'receiving'
            ''')
            conn.commit()

    def setup_merkle_data(self):
        """Generate or load Merkle proofs for all addresses"""
        with sqlite3.connect(DB_PATH) as conn:
            root_hash = conn.execute("SELECT root_hash FROM merkle_proofs LIMIT 1").fetchone()
            if not root_hash:
                self.generate_merkle_proofs()
            else:
                self.merkle_root = root_hash[0]

    def generate_merkle_proofs(self):
        """Generate Merkle proofs for all receiving addresses"""
        with sqlite3.connect(DB_PATH) as conn:
            addresses = [row[0] for row in conn.execute(
                "SELECT address FROM wallets WHERE wallet_type = 'receiving'"
            )]

        leaves = [self.calculate_leaf_hash(addr) for addr in addresses]
        tree = MerkleTree(leaves)
        
        with sqlite3.connect(DB_PATH) as conn:
            for idx, address in enumerate(addresses):
                proof = tree.get_proof(idx)
                conn.execute('''INSERT OR REPLACE INTO merkle_proofs 
                            (wallet_address, proof, root_hash)
                            VALUES (?, ?, ?)''',
                            (address, json.dumps(proof), tree.root))
            conn.commit()
        
        self.merkle_root = tree.root
        self.logger.info(f"Merkle Root: {self.merkle_root}")

    def calculate_leaf_hash(self, address: str) -> str:
        """
        Calculate leaf hash for address using keccak256
        
        Args:
            address: Ethereum address to hash
            
        Returns:
            str: Hex string of the leaf hash
        """
        # Convert address to checksum format
        address = Web3.to_checksum_address(address)
        
        # Encode address as bytes (pad to 32 bytes)
        encoded = bytes.fromhex(address[2:].zfill(64))
        
        # First hash: keccak256(abi.encode(address))
        first_hash = Web3.keccak(encoded)
        
        # Second hash: keccak256(bytes.concat(first_hash))
        leaf_hash = Web3.keccak(first_hash)
        
        return Web3.to_hex(leaf_hash)

    def get_pending_wallets(self):
        """Get wallets that need minting"""
        with sqlite3.connect(DB_PATH) as conn:
            return conn.execute('''
                SELECT w.address, w.private_key
                FROM wallets w
                LEFT JOIN minting_status ms ON w.address = ms.wallet_address
                WHERE w.wallet_type = 'receiving'
                AND (ms.status = 'pending' OR ms.status = 'failed')
                ORDER BY w.address
            ''').fetchall()

    def build_transaction(self, address, nonce):
        """Build EIP-1559 transaction with Merkle proof"""
        return {
            'from': address,
            'to': CONTRACT_ADDRESS,
            'value': self.web3.to_wei(MINT_VALUE, 'ether'),
            'gas': GAS_LIMIT,
            'maxFeePerGas': MAX_FEE,
            'maxPriorityFeePerGas': MAX_PRIORITY_FEE,
            'nonce': nonce,
            'chainId': CHAIN_ID,
            'type': 2,
            'data': self.build_mint_data(address)
        }

    def build_mint_data(self, address):
        """Build mint function data with Merkle proof"""
        with sqlite3.connect(DB_PATH) as conn:
            proof_data = conn.execute(
                "SELECT proof FROM merkle_proofs WHERE wallet_address = ?",
                (address,)
            ).fetchone()
            
        if not proof_data:
            raise ValueError(f"No Merkle proof found for {address}")

        proof = json.loads(proof_data[0])
        qty = 1
        limit = 0
        timestamp = 0
        signature = "0x00"

        contract = self.web3.eth.contract(abi=[{
            "inputs": [
                {"name": "qty", "type": "uint32"},
                {"name": "limit", "type": "uint32"},
                {"name": "proof", "type": "bytes32[]"},
                {"name": "timestamp", "type": "uint256"},
                {"name": "signature", "type": "bytes"}
            ],
            "name": "mint",
            "outputs": [],
            "stateMutability": "payable",
            "type": "function"
        }])

        return contract.encodeABI(
            fn_name="mint",
            args=[qty, limit, proof, timestamp, signature]
        )

    def mint_nfts(self):
        """Main minting function"""
        with sqlite3.connect(DB_PATH) as conn:
            try:
                pending_wallets = self.get_pending_wallets()

                if not pending_wallets:
                    self.logger.info("No pending wallets found")
                    return

                progress_bar = tqdm(pending_wallets, desc="Minting NFTs")

                for address, private_key in progress_bar:
                    if self.should_stop:
                        self.logger.info("Stopped by user")
                        break

                    progress_bar.set_description(f"Minting to {address}")
                    
                    try:
                        # Check balance
                        balance = self.web3.eth.get_balance(address)
                        if balance < self.web3.to_wei(MINT_VALUE, 'ether'):
                            error_msg = f"Insufficient balance: {self.web3.from_wei(balance, 'ether')} ETH"
                            self.update_status(address, 'failed', error_msg=error_msg)
                            continue

                        # Build transaction
                        nonce = self.web3.eth.get_transaction_count(address, 'pending')
                        tx = self.build_transaction(address, nonce)
                        
                        # Sign and send
                        signed_txn = self.web3.eth.account.sign_transaction(tx, private_key)
                        tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
                        
                        self.logger.info(f"Transaction sent: {tx_hash.hex()}")
                        self.update_status(address, 'pending', tx_hash.hex())
                        
                        # Verify transaction
                        success, message = self.verify_transaction(tx_hash, address)
                        
                        if success:
                            self.update_status(address, 'success', tx_hash.hex())
                        else:
                            self.update_status(address, 'failed', tx_hash.hex(), message)
                            if not self.handle_verification_failure():
                                break

                        time.sleep(1)

                    except KeyboardInterrupt:
                        self.logger.info("\nInterrupted by user")
                        break
                    except Exception as e:
                        self.handle_mint_error(address, str(e))
                        if not self.handle_verification_failure():
                            break

                self.show_summary()

            except Exception as e:
                self.logger.error(f"Critical error: {str(e)}")
                raise

    def verify_transaction(self, tx_hash, address):
        """Verify transaction success and NFT receipt"""
        try:
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=60, poll_latency=2)
            
            if receipt.status != 1:
                return False, "Transaction failed on-chain"

            # Wait for state update
            current_block = self.web3.eth.block_number
            while self.web3.eth.block_number < current_block + 2:
                time.sleep(2)

            # Check NFT balance
            balance = self.contract.functions.balanceOf(address).call()
            return (balance > 0, 
                   "NFT received" if balance > 0 else "No NFT received")

        except Exception as e:
            return False, f"Verification error: {str(e)}"

    def update_status(self, address, status, tx_hash=None, error_msg=None):
        """Update minting status in database"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO minting_status 
                (wallet_address, status, tx_hash, error_message, timestamp)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (address, status, tx_hash, error_msg))
            conn.commit()

    def show_summary(self):
        """Show minting summary"""
        with sqlite3.connect(DB_PATH) as conn:
            summary = conn.execute('''
                SELECT status, COUNT(*) FROM minting_status GROUP BY status
            ''').fetchall()
            
        self.logger.info("\nMinting Summary:")
        for status, count in summary:
            self.logger.info(f"{status.capitalize()}: {count}")

    def handle_verification_failure(self):
        """Prompt user after verification failure"""
        response = input("\nVerification failed. Continue? (y/n): ").lower()
        return response == 'y'

    def handle_mint_error(self, address: str, error: str) -> bool:
        """
        Handle minting errors and update status.

        Args:
            address: Wallet address that encountered error
            error: Error message

        Returns:
            bool: True if should continue minting, False otherwise
        """
        self.logger.error(f"Error minting {address}: {error}")
        self.update_status(address, 'failed', error_msg=str(error))
        return self.handle_verification_failure()

    def setup_contract(self) -> object:
        """
        Setup NFT contract with minimal ABI.

        Returns:
            Contract: Web3 contract instance
        """
        abi = [{
            "inputs": [
                {"name": "account", "type": "address"}
            ],
            "name": "balanceOf",
            "outputs": [{"type": "uint256"}],
            "stateMutability": "view",
            "type": "function"
        }]
        return self.web3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)

def main() -> None:
    """Main entry point for the minting script."""
    parser = argparse.ArgumentParser(description='NFT Minting Script')
    parser.add_argument('--mint', action='store_true', help='Start minting NFTs')
    parser.add_argument('--generate-proofs', action='store_true', help='Generate new Merkle proofs')
    parser.add_argument('--resume', action='store_true', help='Resume minting from last failed transaction')
    args = parser.parse_args()

    minter = NFTMinter()

    try:
        if args.generate_proofs:
            # Force regenerate Merkle proofs
            minter.generate_merkle_proofs()
            print("Merkle proofs generated successfully")
            
        elif args.resume:
            # Clear 'failed' status to retry those transactions
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    UPDATE minting_status 
                    SET status = 'pending', 
                        error_message = NULL 
                    WHERE status = 'failed'
                """)
                conn.commit()
            print("Reset failed transactions to pending")
            minter.mint_nfts()
            
        elif args.mint:
            minter.mint_nfts()
            
    except KeyboardInterrupt:
        print("\nStopping process...")
        minter.should_stop = True
    except Exception as e:
        print(f"Critical error: {e}")
        raise

if __name__ == "__main__":
    main()