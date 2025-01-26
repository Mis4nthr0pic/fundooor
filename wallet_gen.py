# wallet_gen.py
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

from web3 import Web3
from eth_account import Account
import csv
import argparse
from pathlib import Path

# Network Configuration
RPC_ENDPOINT = "http://localhost:8545"
CHAIN_ID = 31337

# File Paths
FUNDING_CSV = "funding.csv"
RECEIVING_CSV = "receiving.csv"

# Wallet Configuration
FUNDING_AMOUNT = 10
NUMBER_OF_FUNDING_WALLETS = 30
NUMBER_OF_RECEIVING_WALLETS = 1000

# Anvil Test Account
ANVIL_ACCOUNT = {
    "address": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
    "key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
}

def generate_wallets():
    web3 = Web3(Web3.HTTPProvider(RPC_ENDPOINT))
    if not web3.is_connected():
        raise Exception("Failed to connect to Anvil")

    # Generate funding wallets
    funding_wallets = []
    with open(FUNDING_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['address', 'private_key'])
        for _ in range(NUMBER_OF_FUNDING_WALLETS):
            account = Account.create()
            writer.writerow([account.address, account.key.hex()])
            funding_wallets.append(account.address)

    # Fund each wallet
    for address in funding_wallets:
        tx = {
            'to': address,
            'value': web3.to_wei(FUNDING_AMOUNT, 'ether'),
            'gas': 21000,
            'gasPrice': web3.eth.gas_price,
            'nonce': web3.eth.get_transaction_count(ANVIL_ACCOUNT["address"]),
            'chainId': CHAIN_ID
        }
        signed = web3.eth.account.sign_transaction(tx, ANVIL_ACCOUNT["key"])
        web3.eth.send_raw_transaction(signed.rawTransaction)
        print(f"Funded {address}")

    # Generate receiving wallets
    with open(RECEIVING_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['address', 'private_key'])
        for _ in range(NUMBER_OF_RECEIVING_WALLETS):
            account = Account.create()
            writer.writerow([account.address, account.key.hex()])
    
    print(f"Generated {NUMBER_OF_FUNDING_WALLETS} funding wallets")
    print(f"Generated {NUMBER_OF_RECEIVING_WALLETS} receiving wallets")

def check_balances(wallet_type):
    web3 = Web3(Web3.HTTPProvider(RPC_ENDPOINT))
    csv_file = FUNDING_CSV if wallet_type == 'funding' else RECEIVING_CSV
    
    print(f"\nChecking {wallet_type} wallets:")
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        total_eth = 0
        wallet_count = 0
        funded_count = 0
        
        for row in reader:
            address = row[0]  # First column is address
            private_key = row[1]  # Second column is private key
            balance = web3.from_wei(web3.eth.get_balance(address), 'ether')
            total_eth += balance
            wallet_count += 1
            
            if balance > 0:
                funded_count += 1
                print(f"\nAddress: {address}")
                print(f"Private Key: {private_key}")
                print(f"Balance: {balance} ETH")
                
        print(f"\nSummary:")
        print(f"Total ETH: {total_eth}")
        print(f"Wallets checked: {wallet_count}")
        if wallet_type == 'receiving':
            print(f"Funded wallets: {funded_count}/{wallet_count}")

def main():
    parser = argparse.ArgumentParser(description='Wallet Generator and Balance Checker')
    parser.add_argument('--generate', action='store_true', help='Generate and fund wallets')
    parser.add_argument('--check-balance', choices=['funding', 'receiving'], 
                       help='Check wallet balances (funding or receiving)')
    
    args = parser.parse_args()
    
    if args.generate:
        generate_wallets()
    elif args.check_balance:
        check_balances(args.check_balance)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()