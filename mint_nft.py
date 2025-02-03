from web3 import Web3
import os
from dotenv import load_dotenv
import json
import time
from datetime import datetime
import sqlite3
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Network Configuration
RPC_URL = "https://api.mainnet.abs.xyz"
CHAIN_ID = int("2741")

# Contract Configuration
CONTRACT_ADDRESS = Web3.to_checksum_address('0xe501994195b9951413411395ed1921a88eff694e')
MINT_VALUE = 0.00033  # ETHGAS_LIMIT = 500000
GAS_PRICE = Web3.to_wei(float(os.getenv('GAS_PRICE', '0.04525')), 'gwei')
HEX_DATA = os.getenv('HEX_DATA', '0x')

# Initialize Web3
web3 = Web3(Web3.HTTPProvider(RPC_URL))

class NFTMinter:
    def __init__(self, db_path: str = "distribution.db"):
        self.db_path = db_path
        self.web3 = web3
        if not self.web3.is_connected():
            raise Exception("Failed to connect to RPC endpoint")
        
        print(f"Initialized minter, connected to RPC")
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the minting_status table if it doesn't exist"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS minting_status (
                wallet_address TEXT PRIMARY KEY,
                status TEXT DEFAULT 'pending',
                tx_hash TEXT,
                error_message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Initialize minting_status with unprocessed wallets if empty
        cursor.execute("SELECT COUNT(*) FROM minting_status")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT OR IGNORE INTO minting_status (wallet_address, status)
                SELECT address, 'pending'
                FROM wallets
                WHERE wallet_type = 'receiving'
            """)
        
        conn.commit()
        conn.close()

    def get_db_connection(self):
        """Create a database connection"""
        return sqlite3.connect(self.db_path)

    def process_wallet(self, wallet: Dict[str, Any]):
        """Process a single wallet"""
        try:
            # Create account from wallet's private key
            account = self.web3.eth.account.from_key(wallet['private_key'])
            
            # Your minting logic here using the wallet's own private key
            # ... existing minting code ...
            
            # Update status on success
            self.update_wallet_status(wallet['address'], tx_hash)
            
        except Exception as e:
            print(f"Error processing wallet {wallet['address']}: {str(e)}")
            self.update_wallet_status(wallet['address'], None, str(e))  # Mark as failed
            return False
        
        return True

    def load_wallets_from_db(self) -> List[Dict[str, Any]]:
        """Load pending wallets from minting_status with their private keys"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT m.wallet_address, w.private_key
            FROM minting_status m
            JOIN wallets w ON w.address = m.wallet_address
            WHERE m.status = 'pending'
            AND w.wallet_type = 'receiving'
            ORDER BY m.timestamp ASC
        """
        
        cursor.execute(query)
        wallets = [{
            'address': row[0],
            'private_key': row[1]
        } for row in cursor.fetchall()]
        
        print(f"\nFound {len(wallets)} pending wallets to process:")
        for wallet in wallets:
            print(f"- {wallet['address']}")
        
        conn.close()
        return wallets

    def update_wallet_status(self, wallet_address: str, tx_hash: str = None, error_message: str = None):
        """Update wallet processing status in minting_status"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        status = 'success' if tx_hash else 'failed' if error_message else 'pending'
        
        cursor.execute("""
            UPDATE minting_status 
            SET status = ?,
                tx_hash = ?,
                error_message = ?,
                timestamp = CURRENT_TIMESTAMP
            WHERE wallet_address = ?
        """, (status, tx_hash, error_message, wallet_address))
        
        conn.commit()
        conn.close()

    def get_failed_transactions(self) -> List[Dict[str, Any]]:
        """Get list of failed transactions from minting_status"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT m.wallet_address, w.private_key, m.error_message
            FROM minting_status m
            JOIN wallets w ON w.address = m.wallet_address
            WHERE m.status = 'failed'
            AND w.wallet_type = 'receiving'
            ORDER BY m.timestamp DESC
        """)
        
        failed = [{
            'address': row[0],
            'private_key': row[1],
            'error_message': row[2]
        } for row in cursor.fetchall()]
        
        conn.close()
        return failed

    def mint_nfts(self, start_from_last: bool = True, retry_failed: bool = False):
        """Process NFT minting for wallets"""
        if retry_failed:
            wallets = self.get_failed_transactions()
            print(f"Retrying {len(wallets)} failed transactions")
        else:
            if start_from_last:
                # Get last successfully processed wallet address
                conn = self.get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT wallet_address 
                    FROM minting_status 
                    WHERE status = 'success' 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                result = cursor.fetchone()
                last_address = result[0] if result else ''
                conn.close()
                print(f"Starting after wallet: {last_address}")
            
            wallets = self.load_wallets_from_db()

        if not wallets:
            print("No wallets found to process!")
            return

        success_count = 0
        fail_count = 0

        for wallet in wallets:
            try:
                print(f"\nProcessing wallet: {wallet['address']}")
                print(f"Private key exists: {'private_key' in wallet}")
                
                # Create account from wallet's private key
                try:
                    account = self.web3.eth.account.from_key(wallet['private_key'])
                    print(f"Account created successfully: {account.address}")
                except Exception as e:
                    print(f"Failed to create account: {str(e)}")
                    raise e
                
                # Get the nonce for the transaction
                try:
                    nonce = self.web3.eth.get_transaction_count(wallet['address'], 'pending')
                    print(f"Got nonce: {nonce}")
                except Exception as e:
                    print(f"Failed to get nonce: {str(e)}")
                    raise e
                
                # Prepare the transaction
                transaction = {
                    'from': wallet['address'],
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
                
                print(f"Transaction prepared: {transaction}")

                # Sign and send the transaction
                try:
                    signed_txn = account.sign_transaction(transaction)
                    print("Transaction signed successfully")
                    
                    tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
                    tx_hash_hex = tx_hash.hex()
                    print(f"Transaction sent: {tx_hash_hex}")
                    
                    # Wait for transaction receipt
                    receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash_hex, timeout=120)
                    print(f"Transaction receipt received: {receipt}")
                    
                    if receipt['status'] == 1:
                        print(f"✅ Transaction successful for {wallet['address']}")
                        print(f"Transaction hash: {tx_hash_hex}")
                        self.update_wallet_status(wallet['address'], tx_hash_hex)
                        success_count += 1
                    else:
                        error_msg = f"Transaction failed - reverted by network. Hash: {tx_hash_hex}"
                        print(f"❌ {error_msg}")
                        self.update_wallet_status(wallet['address'], None, error_msg)
                        fail_count += 1
                        
                except Exception as tx_error:
                    tx_hash_str = f" Hash: {tx_hash_hex}" if 'tx_hash_hex' in locals() else ""
                    error_msg = f"Transaction error: {str(tx_error)}.{tx_hash_str}"
                    print(f"❌ {error_msg}")
                    self.update_wallet_status(wallet['address'], None, error_msg)
                    fail_count += 1

            except Exception as e:
                error_msg = f"Processing error: {str(e)}"
                print(f"❌ {error_msg}")
                self.update_wallet_status(wallet['address'], None, error_msg)
                fail_count += 1

        print("\nMinting Summary:")
        print(f"Total processed: {len(wallets)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {fail_count}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NFT Minting CLI')
    parser.add_argument('--mint', action='store_true', help='Start minting process')
    parser.add_argument('--restart', action='store_true', help='Restart minting from the beginning')
    parser.add_argument('--retry-failed', action='store_true', help='Retry all failed transactions')
    parser.add_argument('--status', action='store_true', help='Show minting status and progress')
    parser.add_argument('--recent-tx', action='store_true', help='Show recent transactions')
    parser.add_argument('--failed-list', action='store_true', help='List all failed transactions')
    
    args = parser.parse_args()
    
    minter = NFTMinter()
    
    if args.mint:
        print("Starting minting process")
        minter.mint_nfts()

    elif args.restart:
        print("Restarting minting from the beginning")
        minter.mint_nfts(start_from_last=False)

    elif args.retry_failed:
        print("Retrying failed transactions")
        minter.mint_nfts(retry_failed=True)

    elif args.status:
        conn = minter.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending
            FROM minting_status
        """)
        stats = cursor.fetchone()
        conn.close()

        print(f"""
Minting Status:
Total Wallets: {stats[0]}
Successful: {stats[1]}
Failed: {stats[2]}
Pending: {stats[3]}
        """)

    elif args.recent_tx:
        conn = minter.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT wallet_address, tx_hash, timestamp 
            FROM minting_status 
            WHERE tx_hash IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        recent_txs = cursor.fetchall()
        conn.close()

        print("\nRecent Transactions:")
        for tx in recent_txs:
            print(f"\nWallet: {tx[0]}")
            print(f"TX Hash: {tx[1]}")
            print(f"Processed: {tx[2]}")
            print("-" * 50)

    elif args.failed_list:
        failed = minter.get_failed_transactions()
        print(f"\nFailed Transactions ({len(failed)}):")
        for tx in failed:
            print(f"\nWallet: {tx['address']}")
            print(f"Error: {tx['error_message']}")
            print("-" * 50)

    else:
        parser.print_help()
