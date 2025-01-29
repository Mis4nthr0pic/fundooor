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
        # Load environment variables
        load_dotenv()
        
        # Initialize Web3 and contract
        self.rpc_url = os.getenv('RPC_URL')
        self.contract_address = os.getenv('NFT_CONTRACT_ADDRESS')
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)

        # Check RPC_URL
        if not self.rpc_url:
            self.logger.error("RPC_URL not found in environment variables")
            raise ValueError("RPC_URL is required")
            
        # Initialize Web3
        self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))
        if not self.web3.is_connected():
            raise Exception("Failed to connect to Web3")
            
        # Check NFT_CONTRACT_ADDRESS
        if not self.contract_address:
            self.logger.error("NFT_CONTRACT_ADDRESS not found in environment variables")
            raise ValueError("NFT_CONTRACT_ADDRESS is required")
            
        # Load contract ABI from file
        try:
            with open('contract_abi.json', 'r') as f:
                contract_abi = json.load(f)
        except FileNotFoundError:
            self.logger.error("contract_abi.json not found")
            raise FileNotFoundError("contract_abi.json is required")
            
        # Initialize contract
        self.contract = self.web3.eth.contract(
            address=self.web3.to_checksum_address(self.contract_address),
            abi=contract_abi
        )

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
        """Mint NFTs for all pending addresses"""
        with sqlite3.connect('distribution.db') as conn:
            cursor = conn.execute("""
                SELECT address, private_key 
                FROM wallets 
                WHERE wallet_type = 'receiving'
            """)
            
            total_addresses = cursor.fetchall()
            if not total_addresses:
                self.logger.info("No addresses to mint")
                return

            # Check if we have proofs, if not generate them
            proof_count = conn.execute('SELECT COUNT(*) FROM merkle_proofs').fetchone()[0]
            if proof_count == 0:
                self.logger.info("No proofs found. Generating new proofs...")
                self.generate_proofs()

            for idx, (address, private_key) in enumerate(total_addresses):
                try:
                    # Get proof from merkle_proofs table
                    proof_result = conn.execute(
                        'SELECT proof FROM merkle_proofs WHERE wallet_address = ?',
                        (address,)
                    ).fetchone()

                    if not proof_result:
                        self.logger.info(f"No proof found for {address}. Generating new proof...")
                        cursor = conn.execute(
                            "SELECT DISTINCT address FROM wallets WHERE wallet_type = 'receiving'"
                        )
                        all_addresses = [row[0] for row in cursor.fetchall()]
                        
                        self.merkle_tree = MerkleTree(all_addresses)
                        root = self.merkle_tree.root
                        
                        proof = self.merkle_tree.get_proof(address)
                        conn.execute(
                            'INSERT INTO merkle_proofs (wallet_address, proof, root_hash) VALUES (?, ?, ?)',
                            (address, json.dumps(proof), root)
                        )
                        conn.commit()
                        
                        proof_result = (json.dumps(proof),)

                    proof = json.loads(proof_result[0])
                    
                    # Get nonce for this specific address
                    nonce = self.web3.eth.get_transaction_count(address)
                    
                    # Build transaction
                    tx = self.contract.functions.mint(
                        1,  # qty
                        0,  # limit
                        proof,
                        0,  # timestamp
                        "0x00"  # signature
                    ).build_transaction({
                        'from': address,
                        'gas': 200000,  # Adjust gas as needed
                        'gasPrice': self.web3.eth.gas_price,
                        'nonce': nonce
                    })

                    # Sign and send transaction
                    signed_tx = self.web3.eth.account.sign_transaction(tx, private_key)
                    tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                    
                    self.logger.info(f"Processing {address} ({idx + 1}/{len(total_addresses)})")
                    self.logger.info(f"Transaction hash: {tx_hash.hex()}")
                    
                    # Wait for transaction to be mined
                    receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                    if receipt['status'] != 1:
                        raise Exception("Transaction failed")
                    
                    # Add delay between transactions
                    time.sleep(5)  # 5 second delay between transactions
                    
                except Exception as e:
                    self.logger.error(f"Error processing {address}: {e}")
                    response = input("Error occurred. Continue? (y/n): ")
                    if response.lower() != 'y':
                        break

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

def main():
    """Main entry point for the minting script."""
    parser = argparse.ArgumentParser(description='NFT Minting Script')
    # Add all possible arguments
    parser.add_argument('--generate-proofs', action='store_true', help='Generate new Merkle proofs')
    parser.add_argument('--mint', action='store_true', help='Start minting NFTs')
    parser.add_argument('--resume', action='store_true', help='Resume minting from last failed transaction')
    args = parser.parse_args()

    minter = NFTMinter()

    try:
        if args.generate_proofs:
            root = minter.generate_proofs()
            print(f"Successfully generated new proofs. Merkle root: {root}")
        elif args.mint:
            # Add minting logic here
            minter.mint_nfts()  # Make sure this method exists in NFTMinter class
        elif args.resume:
            # Clear 'failed' status to retry those transactions
            with sqlite3.connect('distribution.db') as conn:
                conn.execute("""
                    UPDATE minting_status 
                    SET status = 'pending', 
                        error_message = NULL 
                    WHERE status = 'failed'
                """)
                conn.commit()
            print("Reset failed transactions to pending")
            minter.mint_nfts()
        else:
            # If no arguments provided, show help
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\nStopping process...")
        minter.should_stop = True
    except Exception as e:
        print(f"Critical error: {e}")
        raise

if __name__ == "__main__":
    main()