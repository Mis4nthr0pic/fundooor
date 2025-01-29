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
CONTRACT_ADDRESS = Web3.to_checksum_address('0xe501994195b9951413411395ed1921a88eff694e')
MINT_VALUE = 0.00033  # ETH value being sent ($1.02)
GAS_LIMIT = 511351  # Gas limit
GAS_USED = 268542   # Actual gas used was 52.52% of limit
BASE_FEE = Web3.to_wei(0.04525, 'gwei')  # Base fee: 0.04525 Gwei
MAX_PRIORITY_FEE = Web3.to_wei(0.04525, 'gwei')  # Max Priority fee: 0.04525 Gwei
MAX_FEE = Web3.to_wei(0.04525, 'gwei')  # Max fee: 0.04525 Gwei
HEX_DATA = "0xb971b4c40000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000a000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000260000000000000000000000000000000000000000000000000000000000000000d7c64cd16bdd8b30eb9231534fdf84de963f86d4016edd91f93f6f0c6a3232dd6a4890148d3f0bacf63abcf240106da530f26a083b7a447bce25d32cff8509cab036d5fcbf331b1ed87cdb8dcc083424eca9236f3a9a5c6fb974e53acd40eb8929ec5fda44757b3df0bbffbeef9b6fb5fdf01ac93d12df2f578871ded34bc1b1c774f1c4e42f0484f107faff26202e9f2be3e1eeb0c487fb8d6598d789271387df77d8fc34d5ea97ee88440c857dc717e58b05ae3e925a4bb26198c10ca3a4ae543e30eae59c297ac961f4a65d50ad85168952853ec71206c0a2be6bf4bb539b0c31119489e7c15128fbb7a8cac334e11a6d8a0325819a1b96ec07477578f8b10a042eb4a5d5d63b9f994ef67d6eb5a7719888e3a0d732d354513c4763444666b55b2b88b47ded8a9ab4d8fc62cfe9d9aa755e9b1bd9d8562aac37b87d8d380a9a8f80182ad3763e208748094ab1d13689d8183c0be0e2d8e7f635563c08f8a288ca3b62d10b78e80c6a9753f9eefc4906ab70e0573cee9ef7a41536b19156ab95f5326affaa037ed35a1eb3b59119e31a7b78cb2792e4860fda048193e0b2b3f00000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000"

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

    def build_transaction(self, address, nonce):
        """Build transaction with exact gas parameters from successful tx"""
        return {
            'from': address,
            'to': CONTRACT_ADDRESS,
            'value': self.web3.to_wei(MINT_VALUE, 'ether'),  # 0.00033 ETH
            'gas': GAS_LIMIT,
            'maxFeePerGas': MAX_FEE,
            'maxPriorityFeePerGas': MAX_PRIORITY_FEE,
            'nonce': nonce,
            'chainId': CHAIN_ID,
            'type': 2,  # EIP-1559 transaction
            'data': HEX_DATA
        }

    def calculate_leaf_hash(self, address: str) -> str:
        """Calculate leaf hash for address the same way contract does:
        keccak256(bytes.concat(keccak256(abi.encode(to))))"""
        
        # First hash: keccak256(abi.encode(address))
        address = address.lower()
        encoded = bytes.fromhex(address[2:].zfill(64))
        first_hash = Web3.keccak(encoded)
        
        # Second hash: keccak256(bytes.concat(first_hash))
        leaf_hash = Web3.keccak(first_hash)
        
        return "0x" + leaf_hash.hex()

    def build_mint_data(self, address):
        """Build mint function data with calculated leaf hash"""
        qty = 1
        limit = 0
        
        # Calculate leaf hash for the address
        leaf_hash = self.calculate_leaf_hash(address)
        self.logger.info(f"Leaf hash for {address}: {leaf_hash}")
        
        # For now, use empty proof until we can calculate the full proof
        proof = []
        timestamp = 0
        signature = "0x00"

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
                        transaction = self.build_transaction(address, nonce)

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
