from web3 import Web3
import os
from dotenv import load_dotenv
import time
import sys

# Load environment variables
load_dotenv()

# Network configuration
RPC_URL = "https://api.mainnet.abs.xyz"
CHAIN_ID = int("2741")

# Connect to the network
web3 = Web3(Web3.HTTPProvider(RPC_URL))

# Contract and wallet configuration
CONTRACT_ADDRESS = Web3.to_checksum_address('0xe501994195b9951413411395ed1921a88eff694e')
MINT_VALUE = 0.00033  # ETH
GAS_LIMIT = 500000  # Exact gas limit from successful tx
PRIVATE_KEY = '89ac1f88763fdb1ae047e0da1ceb602b92dd177cb1d9ec63f52cd744cd952690'

# Gas prices (all set to 0.04525 Gwei as per successful tx)
GAS_PRICE = web3.to_wei(0.04525, 'gwei')

# Exact hex data
HEX_DATA = "0xdf51e12200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000800000000000000000000000000000000000000000000000000000000000000000"

def main():
    if not web3.is_connected():
        raise Exception("Failed to connect to RPC endpoint")
    
    account = web3.eth.account.from_key(PRIVATE_KEY)
    print(f"Using account: {account.address}")
    
    balance = web3.eth.get_balance(account.address)
    balance_in_eth = web3.from_wei(balance, 'ether')
    print(f"\nWallet balance: {balance_in_eth} ETH")
    
    # Get current nonce
    nonce = web3.eth.get_transaction_count(account.address, 'pending')
    
    # Build EIP-1559 transaction with exact gas parameters
    transaction = {
        'from': account.address,
        'to': CONTRACT_ADDRESS,
        'value': web3.to_wei(MINT_VALUE, 'ether'),
        'gas': GAS_LIMIT,
        'maxFeePerGas': GAS_PRICE,
        'maxPriorityFeePerGas': GAS_PRICE,  # Same as max fee
        'nonce': nonce,
        'chainId': CHAIN_ID,
        'type': 2,  # EIP-1559
        'data': HEX_DATA
    }
    
    try:
        # Sign and send transaction
        signed_txn = web3.eth.account.sign_transaction(transaction, PRIVATE_KEY)
        tx_hash = web3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        print(f"\nTransaction sent! Hash: {tx_hash.hex()}")
        print(f"Max fee per gas: {web3.from_wei(GAS_PRICE, 'gwei')} Gwei")
        print(f"Priority fee: {web3.from_wei(GAS_PRICE, 'gwei')} Gwei")
        
        # Wait for transaction receipt
        print("\nWaiting for confirmation...")
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Transaction status: {'Success' if receipt['status'] == 1 else 'Failed'}")
        print(f"Gas used: {receipt['gasUsed']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nDetailed transaction:")
        for key, value in transaction.items():
            print(f"{key}: {value}")
        
    # Show final balance
    final_balance = web3.eth.get_balance(account.address)
    final_balance_eth = web3.from_wei(final_balance, 'ether')
    print(f"\nFinal wallet balance: {final_balance_eth} ETH")

if __name__ == "__main__":
    main()