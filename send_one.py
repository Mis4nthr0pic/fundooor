from web3 import Web3
from eth_account import Account
import os

# Network settings
RPC_ENDPOINT = "https://api.mainnet.abs.xyz"
CHAIN_ID = 2741

def send_one_transaction():
    # Initialize web3
    web3 = Web3(Web3.HTTPProvider(RPC_ENDPOINT))
    
    # Transaction parameters
    private_key = "0xf803b9baba3c61039eabd99d8457c47ac1603846aa633ddb06d93f53114b33ee"
    to_address = "0x001417bF92cBf15609DfE201b64F78Df8de383a7"
    amount_eth = 0.00036
    gas_price_gwei = 0.04525
    
    try:
        # Get account from private key
        account = Account.from_key(private_key)
        from_address = account.address
        
        # Get nonce
        nonce = web3.eth.get_transaction_count(from_address, 'latest')
        
        # Create transaction (UPDATED GAS LIMIT)
        transaction = {
            'nonce': nonce,
            'to': to_address,
            'value': web3.to_wei(amount_eth, 'ether'),
            'gas': 200000,  # Increased from 45250 to accommodate network requirements
            'maxFeePerGas': web3.to_wei(gas_price_gwei, 'gwei'),
            'maxPriorityFeePerGas': web3.to_wei(gas_price_gwei, 'gwei'),
            'type': 2,  # EIP-1559 transaction
            'chainId': CHAIN_ID
        }
        
        # Sign and send transaction
        signed_txn = web3.eth.account.sign_transaction(transaction, private_key)
        tx_hash = web3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        print(f"Transaction sent! Hash: {web3.to_hex(tx_hash)}")
        return web3.to_hex(tx_hash)
        
    except Exception as e:
        print(f"Error sending transaction: {str(e)}")
        return None

if __name__ == "__main__":
    send_one_transaction()