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

# Load environment variables
load_dotenv()

# Configuration
RPC_ENDPOINT = os.getenv('RPC_ENDPOINT')
CHAIN_ID = int(os.getenv('CHAIN_ID', '2741'))
DB_PATH = os.getenv('DB_PATH', 'distribution.db')
DB_LOG_PATH = 'minting.log'

# NFT Contract Configuration
CONTRACT_ADDRESS = Web3.to_checksum_address('0x0525ff85e47ff1b9c23ca88737416ea45a1bf9a5')
MINT_VALUE = 0.00036  # ETH
GAS_LIMIT = 639596
GAS_PRICE = Web3.to_wei(0.04525, 'gwei')
HEX_DATA = "0xdf51e12200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000800000000000000000000000000000000000000000000000000000000000000000"

class NFTMinter:
    def __init__(self):
        self.web3 = Web3(Web3.HTTPProvider(RPC_ENDPOINT))
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(DB_LOG_PATH), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def mint_nfts(self):
        """Mint NFTs from receiving wallets using exact parameters"""
        with sqlite3.connect(DB_PATH) as conn:
            try:
                # Get all receiving wallets with their private keys
                receiving_wallets = conn.execute('''
                    SELECT w.address, w.private_key
                    FROM wallets w
                    WHERE w.wallet_type = 'receiving'
                    ORDER BY w.address
                ''').fetchall()

                if not receiving_wallets:
                    self.logger.info("No receiving wallets found for minting")
                    return

                # Initialize progress bar
                progress_bar = tqdm(receiving_wallets, desc="Minting NFTs")

                success_count = 0
                fail_count = 0

                for address, private_key in progress_bar:
                    progress_bar.set_description(f"Minting to {address}")
                    
                    try:
                        # Check wallet balance
                        balance = self.web3.eth.get_balance(address)
                        balance_in_eth = self.web3.from_wei(balance, 'ether')
                        
                        if balance_in_eth < MINT_VALUE:
                            self.logger.error(f"Insufficient balance in {address}: {balance_in_eth} ETH")
                            fail_count += 1
                            continue

                        # Get current nonce
                        nonce = self.web3.eth.get_transaction_count(address, 'pending')
                        
                        # Build EIP-1559 transaction
                        transaction = {
                            'from': address,
                            'to': CONTRACT_ADDRESS,
                            'value': self.web3.to_wei(MINT_VALUE, 'ether'),
                            'gas': GAS_LIMIT,
                            'maxFeePerGas': GAS_PRICE,
                            'maxPriorityFeePerGas': GAS_PRICE,
                            'nonce': nonce,
                            'chainId': CHAIN_ID,
                            'type': 2,  # EIP-1559
                            'data': HEX_DATA
                        }

                        # Sign and send transaction
                        signed_txn = self.web3.eth.account.sign_transaction(transaction, private_key)
                        tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
                        
                        self.logger.info(f"Transaction sent from {address}! Hash: {tx_hash.hex()}")
                        
                        # Wait for transaction receipt
                        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                        
                        if receipt['status'] == 1:
                            self.logger.info(f"Mint successful for {address}")
                            success_count += 1
                        else:
                            self.logger.error(f"Mint failed for {address}")
                            fail_count += 1

                        # Add delay between transactions
                        time.sleep(1)

                    except Exception as e:
                        self.logger.error(f"Error minting for {address}: {str(e)}")
                        fail_count += 1
                        continue

                self.logger.info(f"\nMinting completed:")
                self.logger.info(f"Success: {success_count}")
                self.logger.info(f"Failed: {fail_count}")
                self.logger.info(f"Total attempted: {len(receiving_wallets)}")

            except Exception as e:
                self.logger.error(f"Error in mint_nfts: {str(e)}")
                raise

def main():
    parser = argparse.ArgumentParser(description='NFT Minting System')
    parser.add_argument('--mint', action='store_true', help='Start minting NFTs')
    
    args = parser.parse_args()
    minter = NFTMinter()
    
    if args.mint:
        minter.mint_nfts()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
