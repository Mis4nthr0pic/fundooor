import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import csv
import sqlite3
import logging
import random
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
from eth_account import Account
from web3 import Web3, exceptions
from web3.types import TxParams
from web3.middleware import geth_poa_middleware
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv()

# Network Configuration
RPC_ENDPOINT = os.getenv('RPC_ENDPOINT')
CHAIN_ID = int(os.getenv('CHAIN_ID', '11124'))
print(RPC_ENDPOINT)
# File Paths
DB_PATH = os.getenv('DB_PATH', 'distribution.db')
DB_LOG_PATH = os.getenv('DB_LOG_PATH', 'distribution.log')
FUNDING_CSV = os.getenv('FUNDING_CSV', 'funding.csv')
RECEIVING_CSV = os.getenv('RECEIVING_CSV', 'receiving.csv')

# Distribution Parameters
MIN_SENDS_PER_WALLET = int(os.getenv('MIN_SENDS_PER_WALLET', '25'))
MAX_SENDS_PER_WALLET = int(os.getenv('MAX_SENDS_PER_WALLET', '35'))
DEFAULT_ETH_AMOUNT = float(os.getenv('DEFAULT_ETH_AMOUNT', '0.0009'))
GAS_LIMIT = int(200000)
DEFAULT_GAS_PRICE = int(os.getenv('DEFAULT_GAS_PRICE', '25'))
MAX_TX_RETRIES = int(os.getenv('MAX_TX_RETRIES', '3'))

# Calculate minimum funding balance
GAS_COST_ETH = (GAS_LIMIT * DEFAULT_GAS_PRICE) / 1e18
MIN_FUNDING_BALANCE = DEFAULT_ETH_AMOUNT + GAS_COST_ETH

# Mainnet mode and delays
MAINNET_MODE = os.getenv('MAINNET_MODE', 'false').lower() == 'true'
TX_DELAY = int(os.getenv('TX_DELAY', '1'))
MAINNET_TX_DELAY = int(os.getenv('MAINNET_TX_DELAY', '3'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '50'))
BATCH_PAUSE = int(os.getenv('BATCH_PAUSE', '30'))

# Database constants
DB_TIMEOUT = 30.0  # Add timeout setting

class Distributor:
    def __init__(self):
        self.web3 = Web3(Web3.HTTPProvider(RPC_ENDPOINT))
        self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
        self.setup_logging()
        self.setup_database()
        self.add_pending_wallets_table()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(DB_LOG_PATH), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
            conn.execute("PRAGMA journal_mode=WAL")  # Use Write-Ahead Logging
            conn.execute("PRAGMA busy_timeout=30000")  # Set busy timeout to 30 seconds
            
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS distribution_plan (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    funding_wallet TEXT NOT NULL,
                    total_receivers INTEGER NOT NULL,
                    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'in_progress', 'completed', 'failed')),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS distribution_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id INTEGER,
                    funding_wallet TEXT NOT NULL,
                    receiving_wallet TEXT NOT NULL,
                    amount_eth TEXT NOT NULL,
                    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'sent', 'failed')),
                    tx_hash TEXT,
                    executed_at DATETIME,
                    FOREIGN KEY(plan_id) REFERENCES distribution_plan(id)
                );

                CREATE TABLE IF NOT EXISTS wallets (
                    address TEXT PRIMARY KEY,
                    wallet_type TEXT CHECK(wallet_type IN ('funding', 'receiving')),
                    private_key TEXT NOT NULL,
                    current_balance TEXT DEFAULT '0',
                    last_updated DATETIME
                );

                CREATE TABLE IF NOT EXISTS pending_funding_wallets (
                    address TEXT PRIMARY KEY,
                    current_balance TEXT NOT NULL,
                    eth_needed TEXT NOT NULL,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Error setting up database: {e}")
            raise

    def add_pending_wallets_table(self):
        """Add the pending_funding_wallets table if it doesn't exist"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pending_funding_wallets (
                    address TEXT PRIMARY KEY,
                    current_balance TEXT NOT NULL,
                    eth_needed TEXT NOT NULL,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def import_wallets(self):
        """
        Import funding and receiving wallets from CSV files into the database.
        """
        with sqlite3.connect(DB_PATH) as conn:
            try:
                # Import funding wallets
                with open(FUNDING_CSV, "r") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header
                    for row in reader:
                        if len(row) != 2:
                            self.logger.error(f"Invalid row in funding.csv: {row}")
                            continue

                        address, private_key = row
                        address = str(address).strip()
                        private_key = str(private_key).strip()

                        if not self.web3.is_address(address):
                            self.logger.error(f"Invalid Ethereum address: {address}")
                            continue

                        balance_wei = self.web3.eth.get_balance(address)
                        balance_eth = self.web3.from_wei(balance_wei, "ether")
                        
                        conn.execute('''
                            INSERT OR REPLACE INTO wallets 
                            (address, wallet_type, private_key, current_balance, last_updated)
                            VALUES (?, 'funding', ?, ?, datetime('now'))
                        ''', (address, private_key, str(balance_eth)))  # Store as string
                        self.logger.info(f"Imported funding wallet: {address}")

                # Import receiving wallets and check their balances
                with open(RECEIVING_CSV, "r") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header
                    for row in reader:
                        if len(row) != 2:
                            self.logger.error(f"Invalid row in receiving.csv: {row}")
                            continue

                        address, private_key = row
                        address = str(address).strip()
                        private_key = str(private_key).strip()

                        if not self.web3.is_address(address):
                            self.logger.error(f"Invalid Ethereum address: {address}")
                            continue

                        balance_wei = self.web3.eth.get_balance(address)
                        balance_eth = self.web3.from_wei(balance_wei, "ether")

                        conn.execute('''
                            INSERT OR REPLACE INTO wallets 
                            (address, wallet_type, private_key, current_balance, last_updated)
                            VALUES (?, 'receiving', ?, ?, datetime('now'))
                        ''', (address, private_key, str(balance_eth)))  # Store as string
                        self.logger.info(f"Imported receiving wallet: {address}")

                conn.commit()
                self.logger.info("Wallets imported successfully.")
                
            except FileNotFoundError as e:
                self.logger.error(f"CSV file not found: {e}")
                raise
            except sqlite3.Error as e:
                self.logger.error(f"Database error: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error during wallet import: {e}")
                raise


    def show_distribution_status(self):
        """
        Display the current status of distribution plans and last processed wallet.
        """
        with sqlite3.connect(DB_PATH) as conn:
            try:
                # Get last processed wallet
                last_processed = conn.execute('''
                    SELECT id, funding_wallet, total_receivers, status, created_at
                    FROM distribution_plan 
                    ORDER BY id DESC LIMIT 1
                ''').fetchone()

                if last_processed:
                    print("\nLast Processed Plan:")
                    headers = ['Plan ID', 'Funding Wallet', 'Total Receivers', 'Status', 'Created At']
                    print(tabulate([[
                        last_processed[0],
                        last_processed[1],
                        last_processed[2],
                        last_processed[3],
                        last_processed[4]
                    ]], headers=headers, tablefmt='grid'))

                # Get distribution summary
                summary = conn.execute('''
                    SELECT 
                        status,
                        COUNT(*) as count,
                        SUM(total_receivers) as total_receivers
                    FROM distribution_plan 
                    GROUP BY status
                ''').fetchall()

                if summary:
                    print("\nDistribution Plans Summary:")
                    headers = ['Status', 'Number of Plans', 'Total Receivers']
                    print(tabulate(summary, headers=headers, tablefmt='grid'))
                else:
                    print("\nNo distribution plans found.")

            except sqlite3.Error as e:
                self.logger.error(f"Database error: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error showing status: {e}")
                raise

    def calculate_total_cost_per_wallet(self, eth_amount, min_sends, max_sends):
        """Calculate total cost needed per wallet including gas"""
        gas_cost_wei = DEFAULT_GAS_PRICE * GAS_LIMIT
        gas_cost_eth = float(self.web3.from_wei(gas_cost_wei, 'ether'))
        cost_per_tx = eth_amount + gas_cost_eth
        
        return {
            'min_total_cost': cost_per_tx * min_sends,
            'max_total_cost': cost_per_tx * max_sends,
            'cost_per_tx': cost_per_tx,
            'gas_cost': gas_cost_eth,
            'transfer_amount': eth_amount
        }

    def insert_distribution_plan(self, conn, distribution_plan):
        """Insert distribution plan into database and return plan ID"""
        try:
            # Get unique funding wallets from the distribution plan
            funding_wallets = set(fw for fw, _, _ in distribution_plan)
            if not funding_wallets:
                raise ValueError("No funding wallets in distribution plan")
            
            # Create a new plan with the first funding wallet (we'll track all in tasks)
            cursor = conn.execute('''
                INSERT INTO distribution_plan (funding_wallet, total_receivers, status, created_at)
                VALUES (?, ?, 'pending', datetime('now'))
            ''', (list(funding_wallets)[0], len(distribution_plan)))
            
            plan_id = cursor.lastrowid

            # Insert tasks with required fields
            conn.executemany('''
                INSERT INTO distribution_tasks 
                (plan_id, funding_wallet, receiving_wallet, amount_eth)
                VALUES (?, ?, ?, ?)
            ''', [(plan_id, fw, rw, amount) for fw, rw, amount in distribution_plan])

            conn.commit()
            return plan_id

        except Exception as e:
            self.logger.error(f"Error inserting distribution plan: {str(e)}")
            conn.rollback()
            raise

    def create_distribution_plan(self, eth_amount=None):
        """Create distribution plan with dynamic wallet distribution"""
        eth_amount = eth_amount if eth_amount is not None else float(os.getenv('DEFAULT_ETH_AMOUNT', '0.00033'))
        
        with sqlite3.connect(DB_PATH) as conn:
            try:
                # Get wallet counts
                funding_count = conn.execute('SELECT COUNT(*) FROM wallets WHERE wallet_type = "funding"').fetchone()[0]
                receiving_count = conn.execute('SELECT COUNT(*) FROM wallets WHERE wallet_type = "receiving"').fetchone()[0]
                
                # Calculate dynamic distribution
                min_sends, max_sends = self.calculate_wallet_distribution(receiving_count, funding_count)
                
                # Calculate costs
                costs = self.calculate_total_cost_per_wallet(eth_amount, min_sends, max_sends)
                
                self.logger.info(f"\nDistribution Parameters:")
                self.logger.info(f"Total receiving wallets: {receiving_count}")
                self.logger.info(f"Total funding wallets: {funding_count}")
                self.logger.info(f"Sends per wallet range: {min_sends} to {max_sends}")
                self.logger.info(f"Transfer amount: {costs['transfer_amount']:.6f} ETH x {max_sends} sends = {costs['transfer_amount'] * max_sends:.6f} ETH")
                self.logger.info(f"Gas cost: {costs['gas_cost']:.6f} ETH x {max_sends} sends = {costs['gas_cost'] * max_sends:.6f} ETH")
                self.logger.info(f"Total min cost per wallet: {costs['min_total_cost']:.6f} ETH")
                self.logger.info(f"Total max cost per wallet: {costs['max_total_cost']:.6f} ETH")

                # Get all wallets and shuffle them
                funding_wallets = conn.execute('''
                    SELECT address 
                    FROM wallets 
                    WHERE wallet_type = 'funding'
                    ORDER BY RANDOM()
                ''').fetchall()

                receiving_wallets = conn.execute('''
                    SELECT address 
                    FROM wallets 
                    WHERE wallet_type = 'receiving'
                    ORDER BY RANDOM()
                ''').fetchall()

                funded_wallets = []
                pending_wallets = []
                distribution_plan = []
                
                # Convert receiving wallets to list and shuffle
                receiving_addresses = [w[0] for w in receiving_wallets]
                random.shuffle(receiving_addresses)
                
                # Calculate even distribution
                receivers_per_funder = len(receiving_addresses) // len(funding_wallets)
                extra_receivers = len(receiving_addresses) % len(funding_wallets)
                
                current_receiver_idx = 0
                
                # Distribute receivers across funding wallets
                for idx, (funding_wallet,) in enumerate(funding_wallets):
                    balance_wei = self.web3.eth.get_balance(funding_wallet)
                    balance_eth = float(self.web3.from_wei(balance_wei, 'ether'))
                    
                    # Calculate how many receivers this wallet should handle
                    wallet_receivers = receivers_per_funder
                    if idx < extra_receivers:  # Distribute any remainder
                        wallet_receivers += 1
                        
                    if balance_eth >= costs['cost_per_tx'] * wallet_receivers:
                        # Add wallet's transactions to distribution plan
                        wallet_receivers_list = receiving_addresses[current_receiver_idx:current_receiver_idx + wallet_receivers]
                        for receiver in wallet_receivers_list:
                            distribution_plan.append((funding_wallet, receiver, eth_amount))
                        current_receiver_idx += wallet_receivers
                        funded_wallets.append((funding_wallet, balance_eth, wallet_receivers))
                    else:
                        eth_needed = (costs['cost_per_tx'] * wallet_receivers) - balance_eth
                        pending_wallets.append((funding_wallet, balance_eth, eth_needed))

                if not funded_wallets:
                    self.logger.error(f"No funding wallets with sufficient balance found")
                    self.logger.error(f"Each wallet needs at least {costs['min_total_cost']:.6f} ETH")
                    return

                # Save pending wallets
                conn.execute('DELETE FROM pending_funding_wallets')
                if pending_wallets:
                    conn.executemany('''
                        INSERT INTO pending_funding_wallets (address, current_balance, eth_needed)
                        VALUES (?, ?, ?)
                    ''', [(addr, str(bal), str(needed)) for addr, bal, needed in pending_wallets])

                # Insert distribution plan
                plan_id = self.insert_distribution_plan(conn, distribution_plan)

                # Show distribution summary
                self.logger.info("\nDistribution Plan Created:")
                self.logger.info(f"Plan ID: {plan_id}")
                self.logger.info(f"Total transactions: {len(distribution_plan)}")
                self.logger.info(f"Funding wallets used: {len(funded_wallets)}")
                self.logger.info(f"Receiving wallets covered: {len(distribution_plan)}")
                
                if current_receiver_idx < len(receiving_addresses):
                    remaining = len(receiving_addresses) - current_receiver_idx
                    self.logger.warning(f"\nWarning: {remaining} receiving wallets not covered")
                    self.logger.warning("Consider funding pending wallets or increasing sends per wallet")

                if pending_wallets:
                    self.logger.info(f"\nPending wallets: {len(pending_wallets)}")
                    self.logger.info("Use --check-funding-needed to see details")

                return plan_id

            except Exception as e:
                self.logger.error(f"Error creating distribution plan: {str(e)}")
                raise

    def send_transaction(self, private_key: str, to_address: str, amount_eth: float) -> Optional[str]:
        """
        Send ETH transaction using EIP-1559 parameters that worked
        """
        try:
            account = Account.from_key(private_key)
            from_address = account.address
            
            # Get nonce
            nonce = self.web3.eth.get_transaction_count(from_address, 'latest')
            
            # Create EIP-1559 transaction with working parameters
            transaction = {
                'nonce': nonce,
                'to': to_address,
                'value': self.web3.to_wei(amount_eth, 'ether'),
                'gas': 200000,  # Increased gas limit that worked
                'maxFeePerGas': self.web3.to_wei('0.04525', 'gwei'),
                'maxPriorityFeePerGas': self.web3.to_wei('0.04525', 'gwei'),
                'type': 2,  # EIP-1559 transaction
                'chainId': CHAIN_ID
            }

            # Sign and send transaction
            signed_txn = self.web3.eth.account.sign_transaction(transaction, private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            return self.web3.to_hex(tx_hash)

        except Exception as e:
            self.logger.error(f"Transaction failed: {str(e)}")
            return None

    def execute_distribution(self):
        """Execute pending distribution tasks with rate limiting"""
        with sqlite3.connect(DB_PATH) as conn:
            try:
                tasks = conn.execute('''
                    SELECT dt.id, dt.funding_wallet, dt.receiving_wallet, dt.amount_eth, w.private_key
                    FROM distribution_tasks dt
                    JOIN wallets w ON dt.funding_wallet = w.address
                    WHERE dt.status = 'pending'
                    ORDER BY dt.id
                ''').fetchall()

                if not tasks:
                    self.logger.info("No pending distribution tasks found")
                    return

                self.logger.info(f"Found {len(tasks)} pending tasks to process")

                # Initialize progress bar
                progress_bar = tqdm(tasks, desc="Processing transactions")

                success_count = 0
                fail_count = 0

                for task_id, funding_wallet, receiving_wallet, amount_eth, private_key in progress_bar:
                    retry_count = 0
                    max_retries = 3
                    while retry_count < max_retries:
                        try:
                            # Send transaction
                            tx_hash = self.send_transaction(
                                private_key=private_key,
                                to_address=receiving_wallet,
                                amount_eth=float(amount_eth)
                            )
                            
                            if tx_hash:
                                # Update task status
                                conn.execute('''
                                    UPDATE distribution_tasks 
                                    SET status = 'sent', 
                                        tx_hash = ?,
                                        executed_at = datetime('now') 
                                    WHERE id = ?
                                ''', (tx_hash, task_id))
                                conn.commit()
                                success_count += 1
                                
                                # Add delay between transactions to avoid rate limits
                                time.sleep(1)  # 1 second delay between transactions
                                break
                            
                        except Exception as e:
                            self.logger.error(f"Error on task {task_id}: {str(e)}")
                            retry_count += 1
                            if retry_count < max_retries:
                                self.logger.info(f"Retry {retry_count} for {receiving_wallet}")
                                time.sleep(5)  # 5 second delay before retry
                            else:
                                fail_count += 1
                                conn.execute('''
                                    UPDATE distribution_tasks 
                                    SET status = 'failed',
                                        executed_at = datetime('now') 
                                    WHERE id = ?
                                ''', (task_id,))
                                conn.commit()

                    # Add longer delay every 10 transactions
                    if success_count % 10 == 0 and success_count > 0:
                        self.logger.info("Rate limit pause - waiting 30 seconds...")
                        time.sleep(30)

                self.logger.info(f"\nExecution completed:")
                self.logger.info(f"Successful transactions: {success_count}")
                self.logger.info(f"Failed transactions: {fail_count}")

            except Exception as e:
                self.logger.error(f"Error executing distribution: {str(e)}")
                raise

    def check_wallet_balance(self, address: str) -> float:
        """
        Check ETH balance for a given wallet address.
        
        Args:
            address: Ethereum wallet address
            
        Returns:
            float: Balance in ETH
            
        Raises:
            ValueError: If address is invalid
        """
        try:
            if not self.web3.is_address(address):
                raise ValueError(f"Invalid Ethereum address: {address}")
            
            balance_wei = self.web3.eth.get_balance(address)
            balance_eth = float(self.web3.from_wei(balance_wei, "ether"))
            
            print(f"\nWallet Balance:")
            print(f"Address: {address}")
            print(f"Balance: {balance_eth:.6f} ETH")
            
            return balance_eth
            
        except Exception as e:
            self.logger.error(f"Error checking balance for {address}: {str(e)}")
            raise

    def check_all_receiving_balances(self):
        """
        Check ETH balance for all receiving wallets and display in a table format.
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                # Get all receiving wallets
                receiving_wallets = conn.execute('''
                    SELECT address 
                    FROM wallets 
                    WHERE wallet_type = 'receiving'
                    ORDER BY address
                ''').fetchall()

                if not receiving_wallets:
                    self.logger.error("No receiving wallets found")
                    return

                # Get balances
                balances = []
                total_balance = 0
                for (address,) in tqdm(receiving_wallets, desc="Checking balances"):
                    balance_wei = self.web3.eth.get_balance(address)
                    balance_eth = float(self.web3.from_wei(balance_wei, "ether"))
                    balances.append([address, f"{balance_eth:.6f}"])
                    total_balance += balance_eth

                # Display results
                print("\nReceiving Wallet Balances:")
                headers = ['Address', 'Balance (ETH)']
                print(tabulate(balances, headers=headers, tablefmt='grid'))
                
                # Display summary
                print(f"\nSummary:")
                print(f"Total wallets: {len(receiving_wallets)}")
                print(f"Total balance: {total_balance:.6f} ETH")
                print(f"Average balance: {(total_balance/len(receiving_wallets)):.6f} ETH")

        except Exception as e:
            self.logger.error(f"Error checking receiving wallet balances: {str(e)}")
            raise

    def update_distribution_amount(self, new_amount: float):
        """
        Update the ETH amount for all pending distribution tasks.
        
        Args:
            new_amount: New amount of ETH to send for each pending transaction
        """
        with sqlite3.connect(DB_PATH) as conn:
            try:
                # Check if there are any pending tasks
                pending_count = conn.execute('''
                    SELECT COUNT(*) 
                    FROM distribution_tasks 
                    WHERE status = 'pending'
                ''').fetchone()[0]

                if pending_count == 0:
                    self.logger.info("No pending tasks found to update")
                    return

                # Update all pending tasks with new amount
                conn.execute('''
                    UPDATE distribution_tasks 
                    SET amount_eth = ?
                    WHERE status = 'pending'
                ''', (str(new_amount),))  # Store as string
                
                conn.commit()
                
                self.logger.info(f"Updated {pending_count} pending tasks to new amount: {new_amount} ETH")

                # Show updated distribution status
                self.show_distribution_status()

            except sqlite3.Error as e:
                self.logger.error(f"Database error: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Error updating distribution amount: {str(e)}")
                raise

    def resend_to_all(self, new_amount: float):
        """
        Reset all tasks to pending and update their amount, then execute distribution.
        
        Args:
            new_amount: New amount of ETH to send for each transaction
        """
        with sqlite3.connect(DB_PATH) as conn:
            try:
                # Reset all tasks to pending and update amount
                conn.execute('''
                    UPDATE distribution_tasks 
                    SET status = 'pending',
                        amount_eth = ?,
                        tx_hash = NULL,
                        executed_at = NULL
                ''', (str(new_amount),))

                # Reset all plans to pending
                conn.execute('''
                    UPDATE distribution_plan 
                    SET status = 'pending'
                ''')
                
                conn.commit()
                
                total_tasks = conn.execute('SELECT COUNT(*) FROM distribution_tasks').fetchone()[0]
                self.logger.info(f"Reset {total_tasks} tasks to pending with new amount: {new_amount} ETH")

                # Show updated status
                self.show_distribution_status()

                # Execute the distribution
                print("\nStarting distribution with new amount...")
                self.execute_distribution()

            except sqlite3.Error as e:
                self.logger.error(f"Database error: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Error resending to all wallets: {str(e)}")
                raise

    def check_all_funding_balances(self):
        """
        Check ETH balance for all funding wallets directly from RPC and display in a table format.
        Shows total, used, and available funds.
        """
        try:
            # Read funding wallets directly from CSV
            funding_wallets = []
            with open(FUNDING_CSV, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) == 2:
                        address = row[0].strip()
                        if self.web3.is_address(address):
                            funding_wallets.append((address,))

            if not funding_wallets:
                self.logger.error("No funding wallets found in CSV")
                return

            self.logger.info(f"Found {len(funding_wallets)} funding wallets")

            # Get balances directly from RPC
            balances = []
            total_balance = 0
            total_pending = 0
            
            for (address,) in tqdm(funding_wallets, desc="Checking funding wallets"):
                try:
                    # Debug log before RPC call
                    self.logger.info(f"Checking balance for {address}")
                    
                    # Get current balance directly from RPC
                    balance_wei = self.web3.eth.get_balance(address)
                    
                    # Debug log after RPC call
                    self.logger.info(f"Raw balance in wei: {balance_wei}")
                    
                    balance_eth = float(self.web3.from_wei(balance_wei, "ether"))
                    
                    # Debug log converted balance
                    self.logger.info(f"Converted balance in ETH: {balance_eth}")
                    
                    balances.append([
                        address,
                        f"{balance_eth:f}",  # Use f instead of .18f to show full decimal
                        "0.000000000000000000",
                        f"{balance_eth:f}"
                    ])
                    
                    total_balance += balance_eth

                except Exception as e:
                    self.logger.error(f"Error checking balance for {address}: {str(e)}")
                    continue

            if not balances:
                self.logger.error("No balances could be retrieved")
                return

            # Display results
            print("\nFunding Wallet Balances (From RPC):")
            headers = ['Address', 'Balance (ETH)', 'Pending (ETH)', 'Available (ETH)']
            print(tabulate(balances, headers=headers, tablefmt='grid'))
            
            # Display summary
            print(f"\nSummary:")
            print(f"Total wallets: {len(funding_wallets)}")
            print(f"Total balance: {total_balance:f} ETH")
            print(f"Total pending: {total_pending:f} ETH")
            print(f"Total available: {total_balance:f} ETH")
            print(f"Average balance per wallet: {(total_balance/len(funding_wallets)):f} ETH")

        except Exception as e:
            self.logger.error(f"Error checking funding wallet balances: {str(e)}")
            raise

    def resume_distribution(self, start_wallet: Optional[str] = None):
        """
        Resume distribution from a specific receiving wallet or the last successful transaction.
        """
        with sqlite3.connect(DB_PATH) as conn:
            try:
                if start_wallet is None:
                    # Find the last successful transaction
                    last_success = conn.execute('''
                        SELECT receiving_wallet 
                        FROM distribution_tasks 
                        WHERE status = 'sent'
                        ORDER BY executed_at DESC
                        LIMIT 1
                    ''').fetchone()
                    
                    if last_success:
                        start_wallet = last_success[0]
                        self.logger.info(f"Resuming from last successful transaction to {start_wallet}")
                    else:
                        self.logger.info("No successful transactions found. Starting from beginning.")
                        start_wallet = '0x0000000000000000000000000000000000000000'

                # Get all pending tasks without filtering by start_wallet
                tasks = conn.execute('''
                    SELECT dt.funding_wallet, dt.receiving_wallet, dt.amount_eth,
                           w.private_key
                    FROM distribution_tasks dt
                    JOIN wallets w ON dt.funding_wallet = w.address
                    WHERE dt.status = 'pending'
                    ORDER BY dt.receiving_wallet
                ''').fetchall()

                if not tasks:
                    self.logger.info("No pending tasks found to resume from the specified start_wallet.")
                    return

                self.logger.info(f"Found {len(tasks)} pending tasks to process starting from {start_wallet}")

                # Initialize progress bar without postfix
                progress_bar = tqdm(
                    tasks, 
                    desc="Processing transactions",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                )

                tx_delay = MAINNET_TX_DELAY if MAINNET_MODE else TX_DELAY
                tx_count = 0
                success_count = 0
                fail_count = 0

                # Process transactions
                for funding_wallet, receiving_wallet, amount_eth, private_key in progress_bar:
                    # Update progress description for current transaction
                    progress_bar.set_description(
                        f"From: {funding_wallet} "
                        f"To: {receiving_wallet}"
                    )
                    self.logger.debug(f"Processing transaction from {funding_wallet} to {receiving_wallet} with amount {amount_eth} ETH")

                    # Check if we need a batch pause
                    if MAINNET_MODE and tx_count > 0 and tx_count % BATCH_SIZE == 0:
                        self.logger.info(f"Batch of {BATCH_SIZE} complete. Pausing for {BATCH_PAUSE} seconds...")
                        time.sleep(BATCH_PAUSE)

                    retry_count = 0
                    tx_hash = None

                    while retry_count < MAX_TX_RETRIES and not tx_hash:
                        if retry_count > 0:
                            time.sleep(tx_delay * 2)  # Double delay on retries
                            self.logger.info(f"Retry {retry_count} for {receiving_wallet}")

                        tx_hash = self.send_transaction(
                            private_key=private_key,
                            to_address=receiving_wallet,
                            amount_eth=float(amount_eth)
                        )
                        retry_count += 1

                    if tx_hash:
                        # Update task status
                        conn.execute('''
                            UPDATE distribution_tasks 
                            SET status = 'sent', 
                                tx_hash = ?,
                                executed_at = datetime('now') 
                            WHERE funding_wallet = ? AND receiving_wallet = ?
                        ''', (tx_hash, funding_wallet, receiving_wallet))

                        # Update wallet balances
                        funding_balance_wei = self.web3.eth.get_balance(funding_wallet)
                        receiving_balance_wei = self.web3.eth.get_balance(receiving_wallet)
                        
                        funding_balance_eth = str(self.web3.from_wei(funding_balance_wei, "ether"))
                        receiving_balance_eth = str(self.web3.from_wei(receiving_balance_wei, "ether"))

                        conn.execute('''
                            UPDATE wallets 
                            SET current_balance = ?,
                                last_updated = datetime('now')
                            WHERE address = ?
                        ''', (funding_balance_eth, funding_wallet))

                        conn.execute('''
                            UPDATE wallets 
                            SET current_balance = ?,
                                last_updated = datetime('now')
                            WHERE address = ?
                        ''', (receiving_balance_eth, receiving_wallet))

                        success_count += 1
                        tx_count += 1

                        # Add delay between successful transactions
                        if tx_count < len(tasks):  # Don't delay after last transaction
                            time.sleep(tx_delay)
                    else:
                        conn.execute('''
                            UPDATE distribution_tasks 
                            SET status = 'failed',
                            executed_at = datetime('now') 
                            WHERE funding_wallet = ? AND receiving_wallet = ?
                        ''', (funding_wallet, receiving_wallet))
                        fail_count += 1

                    conn.commit()

                self.logger.info(f"Resume completed: {success_count} successful, {fail_count} failed")
                self.show_distribution_status()

            except Exception as e:
                self.logger.error(f"Error resuming distribution: {str(e)}")
                raise

    def calculate_wallet_distribution(self, total_receivers, total_funders):
        """Calculate min and max sends per wallet based on ratio"""
        # Calculate base number of transactions per funding wallet
        base_tx_per_wallet = total_receivers / total_funders
        
        # Calculate variation (25% of base)
        variation = round(base_tx_per_wallet * 0.25)
        
        # Calculate min and max sends
        min_sends = max(round(base_tx_per_wallet - variation), 1)  # Never go below 1
        max_sends = round(base_tx_per_wallet + variation)
        
        self.logger.info(f"Distribution calculation:")
        self.logger.info(f"Total receivers: {total_receivers}")
        self.logger.info(f"Total funders: {total_funders}")
        self.logger.info(f"Base tx per wallet: {base_tx_per_wallet:.2f}")
        self.logger.info(f"Variation: ±{variation}")
        self.logger.info(f"Send range: {min_sends} to {max_sends}")
        
        return min_sends, max_sends

    def show_underfunded_wallets(self, eth_amount=None):
        """Show wallets that need funding to meet minimum requirements"""
        if eth_amount is None:
            eth_amount = float(os.getenv('DEFAULT_ETH_AMOUNT', '0.00033'))
        
        # Calculate costs for maximum number of transactions
        gas_cost_wei = DEFAULT_GAS_PRICE * GAS_LIMIT
        gas_cost_eth = float(self.web3.from_wei(gas_cost_wei, 'ether'))
        
        # Calculate total required for max sends
        max_sends = int(os.getenv('MAX_SENDS_PER_WALLET', '25'))
        total_eth_per_wallet = (eth_amount + gas_cost_eth) * max_sends
        
        with sqlite3.connect(DB_PATH) as conn:
            try:
                # Get all funding wallets
                funding_wallets = conn.execute('''
                    SELECT address 
                    FROM wallets 
                    WHERE wallet_type = 'funding'
                    ORDER BY address
                ''').fetchall()

                print("\nChecking for underfunded wallets...")
                print(f"Required balance per wallet: {total_eth_per_wallet:.6f} ETH")
                print(f"- Transfer amount: {eth_amount:.6f} ETH x {max_sends} sends = {eth_amount * max_sends:.6f} ETH")
                print(f"- Gas cost: {gas_cost_eth:.6f} ETH x {max_sends} sends = {gas_cost_eth * max_sends:.6f} ETH")
                print(f"- Max sends per wallet: {max_sends}")

                underfunded_table = []
                total_eth_needed = 0

                for (address,) in funding_wallets:
                    balance_wei = self.web3.eth.get_balance(address)
                    balance_eth = float(self.web3.from_wei(balance_wei, 'ether'))
                    
                    if balance_eth < total_eth_per_wallet:
                        eth_needed = total_eth_per_wallet - balance_eth
                        total_eth_needed += eth_needed
                        underfunded_table.append([
                            address,
                            f"{balance_eth:.6f}",
                            f"{eth_needed:.6f}",
                            f"{int(balance_eth / (eth_amount + gas_cost_eth))}"  # Current possible sends
                        ])

                if underfunded_table:
                    print("\nUnderfunded Wallets:")
                    print(tabulate(underfunded_table, 
                                 headers=['Address', 'Current Balance (ETH)', 'ETH Needed', 'Current Possible Sends'], 
                                 tablefmt='grid'))
                    print(f"\nTotal ETH needed: {total_eth_needed:.6f} ETH")
                    print(f"Total wallets needing funds: {len(underfunded_table)}")
                else:
                    print("\nAll wallets have sufficient funding!")

            except Exception as e:
                self.logger.error(f"Error checking underfunded wallets: {str(e)}")
                raise

    def check_receiving_balances(self):
        """Check receiving wallets and requeue those with zero balance"""
        with sqlite3.connect(DB_PATH) as conn:
            try:
                # Get all receiving wallets
                receiving_wallets = conn.execute('''
                    SELECT address 
                    FROM wallets 
                    WHERE wallet_type = 'receiving'
                ''').fetchall()
                
                zero_balance_wallets = []
                total_wallets = len(receiving_wallets)
                
                self.logger.info(f"Checking balances of {total_wallets} receiving wallets...")
                
                # Initialize progress bar
                progress_bar = tqdm(receiving_wallets, desc="Checking balances")
                
                for (wallet_address,) in progress_bar:
                    try:
                        balance = self.web3.eth.get_balance(wallet_address)
                        if balance == 0:
                            zero_balance_wallets.append(wallet_address)
                    except Exception as e:
                        self.logger.error(f"Error checking balance for {wallet_address}: {str(e)}")
                    
                    # Add small delay to avoid rate limits
                    time.sleep(0.1)
                
                # Display results
                if zero_balance_wallets:
                    self.logger.info(f"\nFound {len(zero_balance_wallets)} wallets with zero balance")
                    
                    # Reset status to pending for tasks with zero balance receiving wallets
                    update_count = conn.execute('''
                        UPDATE distribution_tasks 
                        SET status = 'pending',
                            tx_hash = NULL,
                            executed_at = NULL
                        WHERE receiving_wallet IN ({})
                        AND status != 'pending'
                    '''.format(','.join(['?'] * len(zero_balance_wallets))), zero_balance_wallets).rowcount
                    
                    conn.commit()
                    
                    self.logger.info(f"Reset {update_count} tasks to pending status")
                    self.logger.info("Use --resume to process these transactions again")
                    
                    # Show the wallets that will be reprocessed
                    for wallet in zero_balance_wallets:
                        self.logger.info(f"Will reprocess: {wallet}")
                else:
                    self.logger.info("\nNo wallets with zero balance found")
                
                self.logger.info(f"\nTotal wallets checked: {total_wallets}")
                self.logger.info(f"Zero balance wallets to reprocess: {len(zero_balance_wallets)}")
                
                return zero_balance_wallets
                
            except Exception as e:
                self.logger.error(f"Error checking receiving balances: {str(e)}")
                raise

def main():
    parser = argparse.ArgumentParser(description='ETH Distribution System')
    parser.add_argument('--import-wallets', action='store_true', help='Import wallets from CSV files')
    parser.add_argument('--create-plan', action='store_true', help='Create distribution plan')
    parser.add_argument('--execute', action='store_true', help='Execute distribution')
    parser.add_argument('--status', action='store_true', help='Show distribution status')
    parser.add_argument('--check-balance', type=str, help='Check ETH balance for a wallet address')
    parser.add_argument('--check-receiving', action='store_true', help='Check balance of all receiving wallets')
    parser.add_argument('--check-funding', action='store_true', help='Check balance of all funding wallets')
    parser.add_argument('--amount', type=float, help='Amount of ETH to send to each wallet (default: 0.00002)', default=DEFAULT_ETH_AMOUNT)
    parser.add_argument('--update-amount', type=float, help='Update amount for pending distribution tasks')
    parser.add_argument('--resend-all', type=float, help='Reset all tasks and resend with new amount')
    parser.add_argument('--resume', action='store_true', help='Resume from last successful transaction')
    parser.add_argument('--resume-from', type=str, help='Resume from specific receiving wallet address')
    parser.add_argument('--check-funding-needed', action='store_true', help='Show wallets that need additional funding')
    parser.add_argument('--check-amount', type=float, help='Amount of ETH to check against (default: from .env)',
                       default=float(os.getenv('DEFAULT_ETH_AMOUNT', '0.00033')))
    parser.add_argument('--check-receiving-balances', action='store_true', help='Check receiving wallet balances')
    
    args = parser.parse_args()
    distributor = Distributor()
    
    try:
        if args.import_wallets:
            distributor.import_wallets()
        elif args.create_plan:
            distributor.create_distribution_plan(eth_amount=args.amount)
        elif args.execute:
            distributor.execute_distribution()
        elif args.status:
            distributor.show_distribution_status()
        elif args.check_balance:
            distributor.check_wallet_balance(args.check_balance)
        elif args.check_receiving:
            distributor.check_receiving_balances()
        elif args.check_funding:
            distributor.check_all_funding_balances()
        elif args.update_amount:
            distributor.update_distribution_amount(args.update_amount)
        elif args.resend_all:
            distributor.resend_to_all(args.resend_all)
        elif args.resume or args.resume_from:
            distributor.resume_distribution(start_wallet=args.resume_from)
        elif args.check_funding_needed:
            distributor.show_underfunded_wallets(eth_amount=args.check_amount)
        elif args.check_receiving_balances:
            distributor.check_receiving_balances()
        else:
            parser.print_help()
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()