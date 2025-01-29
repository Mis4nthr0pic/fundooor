from web3 import Web3
import os
from dotenv import load_dotenv
import time
import sys
import json

# Load environment variables
load_dotenv()

# Network configuration
RPC_URL = "https://api.mainnet.abs.xyz"
CHAIN_ID = int("2741")

# Connect to the network
web3 = Web3(Web3.HTTPProvider(RPC_URL))

# Contract and wallet configuration
CONTRACT_ADDRESS = Web3.to_checksum_address('0xE501994195b9951413411395ed1921a88eFF694E')
PRIVATE_KEY = '0x32981ef950cbb48851444be6e704336f263ad6e4b589f0f4732aa283d1957aa1'
MINT_VALUE = 0.00033
GAS_LIMIT = 408418
MAX_FEE = Web3.to_wei(0.04525, 'gwei')
MAX_PRIORITY_FEE = Web3.to_wei(0.04525, 'gwei')

# Contract ABI with merkleRoot function
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
    }, {
        "inputs": [],
        "name": "merkleRoot",
        "outputs": [{"type": "bytes32"}],
        "stateMutability": "view",
        "type": "function"
    }]
)

def hash_leaf(address: str, limit: int) -> bytes:
    """Hash leaf as contract does: keccak256(abi.encodePacked(address, limit))"""
    address = Web3.to_checksum_address(address)
    address_bytes = bytes.fromhex(address[2:])  # Remove '0x' and convert to bytes
    limit_bytes = limit.to_bytes(4, 'big')  # uint32 = 4 bytes
    return Web3.keccak(address_bytes + limit_bytes)

def calculate_root(leaves: list) -> bytes:
    """Calculate Merkle root from list of leaves"""
    if not leaves:
        return None
    
    current_layer = leaves
    while len(current_layer) > 1:
        next_layer = []
        for i in range(0, len(current_layer), 2):
            left = current_layer[i]
            right = current_layer[i + 1] if i + 1 < len(current_layer) else left
            if left < right:
                next_layer.append(Web3.keccak(left + right))
            else:
                next_layer.append(Web3.keccak(right + left))
        current_layer = next_layer
    return current_layer[0]

def get_proof(address: str, limit: int, all_addresses: list) -> list:
    """Calculate Merkle proof for an address with its limit"""
    # Sort and hash all address+limit pairs
    leaves = [hash_leaf(addr, limit) for addr in sorted(all_addresses)]
    
    # Find index of target address
    target_leaf = hash_leaf(address, limit)
    try:
        target_index = leaves.index(target_leaf)
    except ValueError:
        raise ValueError(f"Address {address} not found in merkle tree")
    
    # Build layers of the tree
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
    
    # Calculate proof
    proof = []
    index = target_index
    for layer in layers[:-1]:  # Exclude root layer
        if index % 2 == 0:
            if index + 1 < len(layer):
                proof.append(layer[index + 1])
        else:
            proof.append(layer[index - 1])
        index //= 2
    
    return [p.hex() for p in proof]

def get_active_stage_info(contract):
    """Get current active stage info from contract"""
    current_time = int(time.time())
    num_stages = contract.functions.getNumberStages().call()
    
    for stage in range(num_stages):
        stage_info, _, _ = contract.functions.getStageInfo(stage).call()
        if (current_time >= stage_info['startTimeUnixSeconds'] and 
            current_time < stage_info['endTimeUnixSeconds']):
            return stage, stage_info
    
    raise Exception("No active stage found")

def main():
    if not web3.is_connected():
        raise Exception("Failed to connect to RPC endpoint")
    
    account = web3.eth.account.from_key(PRIVATE_KEY)
    print(f"Using account: {account.address}")
    
    # Get active stage and its merkle root
    try:
        active_stage, stage_info = get_active_stage_info(contract)
        token_id = 0  # Specify which token ID you're minting
        merkle_root = stage_info['merkleRoot'][token_id]
        print(f"\nActive stage: {active_stage}")
        print(f"Merkle root for token {token_id}: {merkle_root.hex()}")
    except Exception as e:
        print(f"Error getting stage info: {e}")
        return
    
    # Get all addresses from the database
    import sqlite3
    with sqlite3.connect('distribution.db') as conn:
        cursor = conn.execute('SELECT address FROM wallets WHERE wallet_type = "receiving"')
        all_addresses = [row[0] for row in cursor.fetchall()]
    
    if not all_addresses:
        raise Exception("No addresses found in database")
    
    # Calculate and verify proof with limit
    limit = stage_info['walletLimit'][token_id]  # Get limit from stage info
    try:
        proof = get_proof(account.address, limit, all_addresses)
        print(f"Merkle proof calculated successfully")
        print(f"Proof: {proof}")
    except ValueError as e:
        print(f"Error calculating proof: {e}")
        return
    
    # Rest of the minting code...
    try:
        nonce = web3.eth.get_transaction_count(account.address, 'pending')
        mint_tx = contract.functions.mint(
            token_id,    # tokenId
            1,          # qty
            limit,      # limit from stage info
            proof       # merkle proof
        ).build_transaction({
            'from': account.address,
            'value': web3.to_wei(MINT_VALUE, 'ether'),
            'gas': GAS_LIMIT,
            'maxFeePerGas': MAX_FEE,
            'maxPriorityFeePerGas': MAX_PRIORITY_FEE,
            'nonce': nonce,
            'chainId': CHAIN_ID,
            'type': 2
        })
        
        print("\nTransaction parameters:")
        print(f"Gas limit: {mint_tx['gas']}")
        print(f"Max fee per gas: {web3.from_wei(mint_tx['maxFeePerGas'], 'gwei')} Gwei")
        print(f"Priority fee: {web3.from_wei(mint_tx['maxPriorityFeePerGas'], 'gwei')} Gwei")
        print(f"Value: {web3.from_wei(mint_tx['value'], 'ether')} ETH")
        
        signed_tx = web3.eth.account.sign_transaction(mint_tx, PRIVATE_KEY)
        tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        print(f"\nTransaction sent! Hash: {tx_hash.hex()}")
        
        print("\nWaiting for confirmation...")
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Transaction status: {'Success' if receipt['status'] == 1 else 'Failed'}")
        print(f"Gas used: {receipt['gasUsed']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if 'mint_tx' in locals():
            print("\nDetailed transaction:")
            for key, value in mint_tx.items():
                print(f"{key}: {value}")

if __name__ == "__main__":
    main()
