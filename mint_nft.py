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
CONTRACT_ADDRESS = Web3.to_checksum_address('0xE501994195b9951413411395ed1921a88eFF694E')  # Updated address
MINT_VALUE = 0.00033  # ETH value being sent ($1.03)
GAS_LIMIT = 408418  # From the successful tx
BASE_GAS = Web3.to_wei(0.04525, 'gwei')
MAX_PRIORITY_FEE = Web3.to_wei(0.04525, 'gwei')
MAX_FEE = Web3.to_wei(0.04525, 'gwei')

# Initialize Web3 and contract with the correct mint function ABI
web3 = Web3(Web3.HTTPProvider(RPC_ENDPOINT))
contract = web3.eth.contract(
    address=CONTRACT_ADDRESS,
    abi=[{
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
    }]
)

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

def hash_leaf(address: str) -> bytes:
    """Hash leaf as contract does: keccak256(abi.encodePacked(address, 0))"""
    address = Web3.to_checksum_address(address)
    address_bytes = bytes.fromhex(address[2:])
    limit_bytes = (0).to_bytes(4, 'big')
    packed = address_bytes + limit_bytes
    return Web3.keccak(packed)

def build_merkle_tree(leaves: List[bytes]) -> List[List[bytes]]:
    """Build the Merkle Tree from leaves"""
    layers = [leaves]
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

def get_proof(address: str, leaves: List[bytes], layers: List[List[bytes]]) -> List[str]:
    """Get the Merkle proof for an address"""
    leaf = hash_leaf(address)
    idx = leaves.index(leaf)
    proof = []
    
    for layer in layers[:-1]:
        pair_idx = idx + 1 if idx % 2 == 0 else idx - 1
        if pair_idx < len(layer):
            proof.append(Web3.to_hex(layer[pair_idx]))
        idx = idx // 2
    
    return proof

def generate_proofs():
    """Generate new proofs for all receiving addresses"""
    with sqlite3.connect(DB_PATH) as conn:
        # Get all receiving addresses
        cursor = conn.execute(
            "SELECT address FROM wallets WHERE wallet_type = 'receiving'"
        )
        addresses = [row[0] for row in cursor.fetchall()]
        
        # Create Merkle tree
        leaves = [hash_leaf(addr) for addr in sorted(addresses)]
        layers = build_merkle_tree(leaves)
        root = Web3.to_hex(layers[-1][0])
        
        # Store proofs for each address
        for address in addresses:
            proof = get_proof(address, leaves, layers)
            conn.execute(
                'INSERT INTO merkle_proofs (wallet_address, proof, root_hash) VALUES (?, ?, ?)',
                (address, json.dumps(proof), root)
            )
        
        conn.commit()
        logger.info(f"Generated Merkle root: {root}")
        return root

def setup_database():
    """Setup database tables for tracking mint status"""
    with sqlite3.connect(DB_PATH) as conn:
        # Create mint_status table if it doesn't exist
        conn.execute('''
            CREATE TABLE IF NOT EXISTS mint_status (
                address TEXT PRIMARY KEY,
                status TEXT,  -- 'pending', 'success', 'failed'
                error_message TEXT,
                tx_hash TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

def get_pending_addresses():
    """Get addresses that haven't been processed or failed"""
    with sqlite3.connect(DB_PATH) as conn:
        # Get addresses that are either not in mint_status or failed
        cursor = conn.execute('''
            SELECT w.address, w.private_key 
            FROM wallets w 
            LEFT JOIN mint_status m ON w.address = m.address
            WHERE w.wallet_type = 'receiving'
            AND (m.status IS NULL OR m.status = 'failed')
        ''')
        return cursor.fetchall()

def update_mint_status(address, status, error_message=None, tx_hash=None):
    """Update minting status for an address"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            INSERT OR REPLACE INTO mint_status 
            (address, status, error_message, tx_hash, timestamp)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (address, status, error_message, tx_hash))
        conn.commit()

def mint_nfts():
    """Mint NFTs for all pending addresses"""
    # Setup tracking table
    setup_database()
    
    # Get pending addresses
    pending_addresses = get_pending_addresses()
    if not pending_addresses:
        logger.info("No pending addresses to process")
        return

    logger.info(f"Found {len(pending_addresses)} addresses to process")
    failed_addresses = set()

    for idx, (address, private_key) in enumerate(pending_addresses):
        if address in failed_addresses:
            continue
            
        try:
            # Get proof from merkle_proofs table
            with sqlite3.connect(DB_PATH) as conn:
                proof_result = conn.execute(
                    'SELECT proof FROM merkle_proofs WHERE wallet_address = ?',
                    (address,)
                ).fetchone()

            if not proof_result:
                logger.info(f"No proof found for {address}. Generating new proofs...")
                generate_proofs()
                with sqlite3.connect(DB_PATH) as conn:
                    proof_result = conn.execute(
                        'SELECT proof FROM merkle_proofs WHERE wallet_address = ?',
                        (address,)
                    ).fetchone()

            proof = json.loads(proof_result[0])
            nonce = web3.eth.get_transaction_count(address)

            # Build mint transaction
            mint_tx = contract.functions.mint(
                1, 0, proof, 0, "0x00"
            ).build_transaction({
                'from': address,
                'value': web3.to_wei(MINT_VALUE, 'ether'),
                'gas': GAS_LIMIT,
                'maxFeePerGas': MAX_FEE,
                'maxPriorityFeePerGas': MAX_PRIORITY_FEE,
                'nonce': nonce,
                'chainId': CHAIN_ID,
                'type': 2
            })

            # Log transaction details
            logger.info(f"\nProcessing {address} ({idx + 1}/{len(pending_addresses)})")
            logger.info(f"Gas: {mint_tx['gas']}")
            logger.info(f"Max fee: {web3.from_wei(mint_tx['maxFeePerGas'], 'gwei')} Gwei")
            logger.info(f"Priority fee: {web3.from_wei(mint_tx['maxPriorityFeePerGas'], 'gwei')} Gwei")

            # Sign and send
            signed_tx = web3.eth.account.sign_transaction(mint_tx, private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            logger.info(f"Transaction hash: {tx_hash.hex()}")
            
            # Update status as pending with tx hash
            update_mint_status(address, 'pending', tx_hash=tx_hash.hex())
            
            # Wait for receipt
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            if receipt['status'] != 1:
                raise Exception("Transaction failed")
            
            # Update status as success
            update_mint_status(address, 'success', tx_hash=tx_hash.hex())
            logger.info(f"Success: {address}")
            
            time.sleep(int(os.getenv('TX_DELAY', '1')))
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing {address}: {error_msg}")
            update_mint_status(address, 'failed', error_message=error_msg)
            failed_addresses.add(address)
            continue

    # Print summary
    with sqlite3.connect(DB_PATH) as conn:
        stats = conn.execute('''
            SELECT status, COUNT(*) 
            FROM mint_status 
            GROUP BY status
        ''').fetchall()
        
    logger.info("\nMinting Summary:")
    for status, count in stats:
        logger.info(f"{status.capitalize()}: {count}")

def validate_mint_status(address):
    """
    Validate mint status for an address by checking NFT balance
    Returns: (bool, str) - (success, message)
    """
    try:
        # Add balanceOf to contract ABI if not present
        balance_abi = {
            "inputs": [{"name": "account", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"type": "uint256"}],
            "stateMutability": "view",
            "type": "function"
        }
        
        # Check NFT balance
        balance = contract.functions.balanceOf(address).call()
        
        if balance > 0:
            return True, f"Success: Address has {balance} NFT(s)"
        else:
            return False, "No NFTs found for this address"
            
    except Exception as e:
        return False, f"Error checking balance: {str(e)}"

def check_mint_status():
    """Check and display mint status for all addresses"""
    with sqlite3.connect(DB_PATH) as conn:
        # Get all addresses and their current status
        cursor = conn.execute('''
            SELECT w.address, COALESCE(m.status, 'not_started') as status, 
                   m.tx_hash, m.error_message
            FROM wallets w
            LEFT JOIN mint_status m ON w.address = m.address
            WHERE w.wallet_type = 'receiving'
        ''')
        addresses = cursor.fetchall()

    logger.info("\nChecking mint status for all addresses...")
    
    results = {
        'success': 0,
        'failed': 0,
        'pending': 0,
        'not_started': 0
    }

    for address, db_status, tx_hash, error_msg in addresses:
        success, message = validate_mint_status(address)
        
        logger.info(f"\nAddress: {address}")
        logger.info(f"Database status: {db_status}")
        logger.info(f"Actual status: {'Success' if success else 'Failed'}")
        logger.info(f"Message: {message}")
        
        if tx_hash:
            logger.info(f"Transaction: {tx_hash}")
        if error_msg:
            logger.info(f"Error: {error_msg}")
            
        # Update database with validated status
        new_status = 'success' if success else 'failed'
        if new_status != db_status:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute('''
                    UPDATE mint_status 
                    SET status = ?, 
                        error_message = ?
                    WHERE address = ?
                ''', (new_status, message if not success else None, address))
                conn.commit()
        
        results[db_status] += 1

    # Print summary
    logger.info("\nSummary:")
    logger.info(f"Total addresses: {len(addresses)}")
    logger.info(f"Successful: {results['success']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Pending: {results['pending']}")
    logger.info(f"Not started: {results['not_started']}")

def main():
    parser = argparse.ArgumentParser(description='NFT Minting Script')
    parser.add_argument('--generate-proofs', action='store_true', help='Generate new Merkle proofs')
    parser.add_argument('--mint', action='store_true', help='Start minting NFTs')
    parser.add_argument('--retry-failed', action='store_true', help='Retry failed transactions')
    parser.add_argument('--check-status', action='store_true', help='Check mint status for all addresses')
    args = parser.parse_args()

    try:
        if args.generate_proofs:
            root = generate_proofs()
            print(f"Successfully generated new proofs. Merkle root: {root}")
        elif args.mint:
            mint_nfts()
        elif args.retry_failed:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("UPDATE mint_status SET status = NULL WHERE status = 'failed'")
                conn.commit()
            mint_nfts()
        elif args.check_status:
            check_mint_status()
        else:
            parser.print_help()
            
    except Exception as e:
        logger.error(f"Critical error: {e}")
        raise

if __name__ == "__main__":
    main()