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
from typing import List, Tuple

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
    def __init__(self, leaves: List[str]):
        """Initialize Merkle Tree with list of addresses"""
        # Hash leaves as contract does: keccak256(abi.encodePacked(address, 0))
        self.leaves = [self._hash_leaf(addr) for addr in sorted(leaves)]
        self.layers = self._build_tree()

    def _hash_leaf(self, address: str) -> bytes:
        """Hash leaf as contract does: keccak256(abi.encodePacked(address, 0))"""
        address = Web3.to_checksum_address(address)
        # Manual packing: 20 bytes address + 4 bytes of zero for uint32
        address_bytes = bytes.fromhex(address[2:])  # Remove '0x' and convert to bytes
        limit_bytes = (0).to_bytes(4, 'big')  # 4 bytes of zeros for uint32
        packed = address_bytes + limit_bytes
        return Web3.keccak(packed)

    def _build_tree(self):
        """Build the Merkle Tree from leaves"""
        layers = [self.leaves]
        while len(layers[-1]) > 1:
            layer = []
            for i in range(0, len(layers[-1]), 2):
                left = layers[-1][i]
                right = layers[-1][i + 1] if i + 1 < len(layers[-1]) else left
                if left < right:
                    layer.append(Web3.keccak(left + right))
                else:
                    layer.append(Web3.keccak(right + left))
            layers.append(layer)
        return layers

    def get_proof(self, address: str) -> List[str]:
        """Get the Merkle proof for an address"""
        leaf = self._hash_leaf(address)
        idx = self.leaves.index(leaf)
        proof = []
        
        for layer in self.layers[:-1]:
            pair_idx = idx + 1 if idx % 2 == 0 else idx - 1
            if pair_idx < len(layer):
                proof.append(Web3.to_hex(layer[pair_idx]))
            idx = idx // 2
        
        return proof

    @property
    def root(self) -> str:
        """Get Merkle root"""
        return Web3.to_hex(self.layers[-1][0])

class NFTMinter:
    def __init__(self):
        self.web3 = Web3(Web3.HTTPProvider(RPC_ENDPOINT))
        self.contract = self.setup_contract()
        self.merkle_tree = None
        self.setup_logging()
        self.should_stop = False
        self.setup_database()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(DB_LOG_PATH), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        """Setup or clear the merkle_proofs table"""
        with sqlite3.connect(DB_PATH) as conn:
            # Drop existing merkle_proofs table if it exists
            conn.execute('DROP TABLE IF EXISTS merkle_proofs')
            
            # Create fresh merkle_proofs table
            conn.execute('''
                CREATE TABLE merkle_proofs (
                    wallet_address TEXT PRIMARY KEY,
                    proof TEXT,
                    root_hash TEXT
                )
            ''')
            conn.commit()

    def generate_proofs(self):
        """Generate new proofs for all receiving addresses"""
        with sqlite3.connect(DB_PATH) as conn:
            # Get all receiving addresses
            addresses = [row[0] for row in conn.execute(
                "SELECT address FROM wallets WHERE wallet_type = 'receiving'"
            )]

            # Create Merkle tree
            self.merkle_tree = MerkleTree(addresses)
            root = self.merkle_tree.root
            
            # Store proofs for each address
            for address in addresses:
                proof = self.merkle_tree.get_proof(address)
                conn.execute(
                    'INSERT INTO merkle_proofs (wallet_address, proof, root_hash) VALUES (?, ?, ?)',
                    (address, json.dumps(proof), root)
                )
            
            conn.commit()
            self.logger.info(f"Generated Merkle root: {root}")
            return root

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

    def build_mint_data(self, address: str):
        """Build mint function data with Merkle proof"""
        address = Web3.to_checksum_address(address)
        
        # Get proof from database
        with sqlite3.connect(DB_PATH) as conn:
            result = conn.execute(
                'SELECT proof FROM merkle_proofs WHERE wallet_address = ?',
                (address,)
            ).fetchone()
            
            if not result:
                raise ValueError(f"No proof found for address {address}")
            
            proof = json.loads(result[0])

        # Contract parameters
        qty = 1
        limit = 0  # Always 0 as specified
        timestamp = 0  # Always 0 as specified
        signature = "0x00"  # Always 0x00 as specified

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
    parser.add_argument('--generate-proofs', action='store_true', help='Generate new Merkle proofs')
    args = parser.parse_args()

    minter = NFTMinter()

    try:
        if args.generate_proofs:
            root = minter.generate_proofs()
            print(f"Successfully generated new proofs. Merkle root: {root}")
            
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()