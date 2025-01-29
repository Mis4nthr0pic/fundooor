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
        self.contract = self.setup_contract()
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
        """Setup database tables if they don't exist"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS minting_status (
                    wallet_address TEXT PRIMARY KEY,
                    status TEXT CHECK(status IN ('pending', 'success', 'failed')) DEFAULT 'pending',
                    tx_hash TEXT,
                    error_message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Initialize minting status for new wallets
            conn.execute('''
                INSERT OR IGNORE INTO minting_status (wallet_address, status)
                SELECT address, 'pending'
                FROM wallets
                WHERE wallet_type = 'receiving'
            ''')
            conn.commit()

    def get_pending_wallets(self):
        """Get wallets that still need minting"""
        with sqlite3.connect(DB_PATH) as conn:
            return conn.execute('''
                SELECT w.address, w.private_key
                FROM wallets w
                LEFT JOIN minting_status ms ON w.address = ms.wallet_address
                WHERE w.wallet_type = 'receiving'
                AND (ms.status = 'pending' OR ms.status = 'failed')
                ORDER BY w.address
            ''').fetchall()

    def update_minting_status(self, address, status, tx_hash=None, error_message=None):
        """Update minting status for a wallet"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO minting_status 
                (wallet_address, status, tx_hash, error_message, timestamp)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (address, status, tx_hash, error_message))
            conn.commit()

    def get_minting_summary(self):
        """Get summary of minting progress"""
        with sqlite3.connect(DB_PATH) as conn:
            return conn.execute('''
                SELECT status, COUNT(*) as count
                FROM minting_status
                GROUP BY status
            ''').fetchall()

    def setup_contract(self):
        """Setup NFT contract interface"""
        # Basic ERC721 ABI - add specific functions you need
        abi = [
            {
                "inputs": [{"internalType": "address", "name": "owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        return self.web3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)

    def verify_transaction(self, tx_hash, address):
        """Verify if the transaction was successful and NFT was received"""
        try:
            # Wait for transaction receipt with timeout
            receipt = self.web3.eth.wait_for_transaction_receipt(
                tx_hash, 
                timeout=60,  # 60 second timeout
                poll_latency=2  # Check every 2 seconds
            )
            
            if receipt['status'] != 1:
                return False, "Transaction failed on-chain"

            # Wait a few blocks for state to update
            current_block = self.web3.eth.block_number
            target_block = current_block + 2
            while self.web3.eth.block_number < target_block:
                time.sleep(2)

            # Check NFT balance after transaction
            try:
                balance_after = self.contract.functions.balanceOf(address).call()
                if balance_after > 0:
                    return True, "Transaction successful and NFT received"
                else:
                    return False, "Transaction successful but no NFT received"
            except Exception as e:
                return False, f"Error checking NFT balance: {str(e)}"

        except Exception as e:
            return False, f"Error verifying transaction: {str(e)}"

    def mint_nfts(self):
        """Mint NFTs from receiving wallets using exact parameters"""
        with sqlite3.connect(DB_PATH) as conn:
            try:
                pending_wallets = self.get_pending_wallets()

                if not pending_wallets:
                    self.logger.info("No pending wallets found for minting")
                    return

                progress_bar = tqdm(pending_wallets, desc="Minting NFTs")

                for address, private_key in progress_bar:
                    if self.should_stop:
                        self.logger.info("Minting process stopped by user")
                        break

                    progress_bar.set_description(f"Minting to {address}")
                    
                    try:
                        # Check wallet balance
                        balance = self.web3.eth.get_balance(address)
                        balance_in_eth = self.web3.from_wei(balance, 'ether')
                        
                        if balance_in_eth < MINT_VALUE:
                            error_msg = f"Insufficient balance: {balance_in_eth} ETH"
                            self.logger.error(f"{address}: {error_msg}")
                            self.update_minting_status(address, 'failed', error_message=error_msg)
                            continue

                        # Get current nonce
                        nonce = self.web3.eth.get_transaction_count(address, 'pending')
                        
                        # Build transaction
                        transaction = {
                            'from': address,
                            'to': CONTRACT_ADDRESS,
                            'value': self.web3.to_wei(MINT_VALUE, 'ether'),
                            'gas': GAS_LIMIT,
                            'maxFeePerGas': GAS_PRICE,
                            'maxPriorityFeePerGas': GAS_PRICE,
                            'nonce': nonce,
                            'chainId': CHAIN_ID,
                            'type': 2,
                            'data': HEX_DATA
                        }

                        # Sign and send transaction
                        signed_txn = self.web3.eth.account.sign_transaction(transaction, private_key)
                        tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
                        
                        self.logger.info(f"Transaction sent from {address}! Hash: {tx_hash.hex()}")
                        
                        # Update status to pending with transaction hash
                        self.update_minting_status(address, 'pending', tx_hash.hex())
                        
                        # Verify transaction success
                        success, message = self.verify_transaction(tx_hash, address)
                        
                        if success:
                            self.logger.info(f"Mint verified successful for {address}")
                            self.update_minting_status(address, 'success', tx_hash.hex())
                        else:
                            self.logger.error(f"Mint verification failed for {address}: {message}")
                            self.update_minting_status(
                                address, 
                                'failed', 
                                tx_hash.hex(), 
                                f"Verification failed: {message}"
                            )
                            
                            response = input("\nVerification failed. Continue minting? (y/n): ").lower()
                            if response != 'y':
                                self.logger.info("Minting process stopped by user")
                                break

                        # Add delay between transactions
                        time.sleep(1)

                    except KeyboardInterrupt:
                        self.logger.info("\nMinting process interrupted by user")
                        break
                    except Exception as e:
                        error_msg = str(e)
                        self.logger.error(f"Error minting for {address}: {error_msg}")
                        self.update_minting_status(address, 'failed', error_message=error_msg)
                        
                        response = input("\nError occurred. Continue minting? (y/n): ").lower()
                        if response != 'y':
                            self.logger.info("Minting process stopped by user")
                            break
                        continue

                # Print final summary
                summary = self.get_minting_summary()
                self.logger.info("\nMinting Summary:")
                for status, count in summary:
                    self.logger.info(f"{status.capitalize()}: {count}")

            except Exception as e:
                self.logger.error(f"Error in mint_nfts: {str(e)}")
                raise

def main():
    parser = argparse.ArgumentParser(description='NFT Minting System')
    parser.add_argument('--mint', action='store_true', help='Start minting NFTs')
    parser.add_argument('--status', action='store_true', help='Show minting status')
    parser.add_argument('--failed', action='store_true', help='Show failed transactions')
    parser.add_argument('--retry-failed', action='store_true', help='Retry failed transactions only')
    
    args = parser.parse_args()
    minter = NFTMinter()
    
    if args.mint:
        try:
            minter.mint_nfts()
        except KeyboardInterrupt:
            minter.stop_minting()
    elif args.retry_failed:
        try:
            # Reset status of failed transactions to pending
            with sqlite3.connect(DB_PATH) as conn:
                failed_count = conn.execute('''
                    UPDATE minting_status 
                    SET status = 'pending', 
                        tx_hash = NULL, 
                        error_message = NULL, 
                        timestamp = CURRENT_TIMESTAMP
                    WHERE status = 'failed'
                ''').rowcount
                conn.commit()
                
            if failed_count > 0:
                print(f"\nResetting {failed_count} failed transactions to pending status...")
                minter.mint_nfts()  # This will now process the reset transactions
            else:
                print("\nNo failed transactions found to retry.")
                
        except KeyboardInterrupt:
            minter.stop_minting()
    elif args.status:
        summary = minter.get_minting_summary()
        print("\nMinting Status:")
        for status, count in summary:
            print(f"{status.capitalize()}: {count}")
    elif args.failed:
        with sqlite3.connect(DB_PATH) as conn:
            failed = conn.execute('''
                SELECT wallet_address, tx_hash, error_message, timestamp
                FROM minting_status
                WHERE status = 'failed'
                ORDER BY timestamp DESC
            ''').fetchall()
            
            if not failed:
                print("\nNo failed transactions found.")
            else:
                print(f"\nFound {len(failed)} Failed Transactions:")
                for wallet, tx_hash, error, timestamp in failed:
                    print(f"\nWallet: {wallet}")
                    print(f"Time: {timestamp}")
                    print(f"Error: {error}")
                    if tx_hash:
                        print(f"TX Hash: {tx_hash}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
