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
MINT_VALUE = 0.00033  # ETH value being sent
GAS_LIMIT = 223789  # Adjusted gas limit
GAS_PRICE = Web3.to_wei(0.0000000452, 'ether')  # Adjusted to match the exact fee

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

def mint_nfts():
    """Mint NFTs for all pending addresses"""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("""
            SELECT address, private_key 
            FROM wallets 
            WHERE wallet_type = 'receiving'
        """)
        
        total_addresses = cursor.fetchall()
        if not total_addresses:
            logger.info("No addresses to mint")
            return

        # Check if we have proofs
        proof_count = conn.execute('SELECT COUNT(*) FROM merkle_proofs').fetchone()[0]
        if proof_count == 0:
            logger.info("No proofs found. Generating new proofs...")
            generate_proofs()

        # Keep track of failed addresses
        failed_addresses = set()

        for idx, (address, private_key) in enumerate(total_addresses):
            if address in failed_addresses:
                logger.info(f"Skipping previously failed address {address}")
                continue
                
            try:
                # Get proof
                proof_result = conn.execute(
                    'SELECT proof FROM merkle_proofs WHERE wallet_address = ?',
                    (address,)
                ).fetchone()

                if not proof_result:
                    logger.info(f"No proof found for {address}. Generating new proofs...")
                    generate_proofs()
                    proof_result = conn.execute(
                        'SELECT proof FROM merkle_proofs WHERE wallet_address = ?',
                        (address,)
                    ).fetchone()

                proof = json.loads(proof_result[0])
                nonce = web3.eth.get_transaction_count(address)

                # Build mint transaction with exact fee parameters
                mint_tx = contract.functions.mint(
                    1,      # qty
                    0,      # limit
                    proof,  # merkle proof
                    0,      # timestamp
                    "0x00"  # signature
                ).build_transaction({
                    'from': address,
                    'value': web3.to_wei(MINT_VALUE, 'ether'),
                    'gas': GAS_LIMIT,
                    'gasPrice': GAS_PRICE,  # Using gasPrice instead of maxFeePerGas
                    'nonce': nonce,
                    'chainId': CHAIN_ID,
                    'type': 0  # Legacy transaction type
                })

                # Sign and send
                signed_tx = web3.eth.account.sign_transaction(mint_tx, private_key)
                tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                logger.info(f"Processing {address} ({idx + 1}/{len(total_addresses)})")
                logger.info(f"Transaction hash: {tx_hash.hex()}")
                
                # Wait for transaction
                receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
                if receipt['status'] != 1:
                    raise Exception("Transaction failed")
                
                # Delay between transactions
                delay = int(os.getenv('MAINNET_TX_DELAY' if os.getenv('MAINNET_MODE') == 'true' else 'TX_DELAY'))
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error processing {address}: {e}")
                failed_addresses.add(address)  # Add to failed addresses
                continue  # Skip to next address instead of asking for input
                
        # Print summary of failed addresses at the end
        if failed_addresses:
            logger.info("\nFailed addresses:")
            for addr in failed_addresses:
                logger.info(addr)
            logger.info(f"\nTotal failed: {len(failed_addresses)}")

def main():
    parser = argparse.ArgumentParser(description='NFT Minting Script')
    parser.add_argument('--generate-proofs', action='store_true', help='Generate new Merkle proofs')
    parser.add_argument('--mint', action='store_true', help='Start minting NFTs')
    args = parser.parse_args()

    try:
        if args.generate_proofs:
            root = generate_proofs()
            print(f"Successfully generated new proofs. Merkle root: {root}")
        elif args.mint:
            mint_nfts()
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()