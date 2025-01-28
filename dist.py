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
GAS_LIMIT = int(os.getenv('GAS_LIMIT', '21000'))
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
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Error setting up database: {e}")
            raise

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

    def calculate_funding_requirements(self, eth_amount, min_sends, max_sends):
        """Calculate funding requirements considering send variation"""
        gas_cost_wei = DEFAULT_GAS_PRICE * GAS_LIMIT
        gas_cost_eth = float(self.web3.from_wei(gas_cost_wei, 'ether'))
        cost_per_tx = eth_amount + gas_cost_eth
        
        return {
            'min_balance': cost_per_tx * min_sends,
            'max_balance': cost_per_tx * max_sends,
            'cost_per_tx': cost_per_tx,
            'gas_cost': gas_cost_eth
        }

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
                
                # Calculate funding requirements
                funding_reqs = self.calculate_funding_requirements(eth_amount, min_sends, max_sends)
                
                self.logger.info(f"\nDistribution Parameters:")
                self.logger.info(f"Total receiving wallets: {receiving_count}")
                self.logger.info(f"Total funding wallets: {funding_count}")
                self.logger.info(f"Sends per wallet range: {min_sends} to {max_sends}")
                self.logger.info(f"ETH per transaction: {eth_amount}")
                self.logger.info(f"Gas cost per tx: {funding_reqs['gas_cost']:.6f} ETH")
                self.logger.info(f"Min balance needed: {funding_reqs['min_balance']:.6f} ETH ({min_sends} sends)")
                self.logger.info(f"Max balance needed: {funding_reqs['max_balance']:.6f} ETH ({max_sends} sends)")

                # Get and categorize funding wallets
                funding_wallets = conn.execute('''
                    SELECT address 
                    FROM wallets 
                    WHERE wallet_type = 'funding'
                    ORDER BY RANDOM()
                ''').fetchall()

                funded_wallets = []
                pending_wallets = []

                for (address,) in funding_wallets:
                    balance_wei = self.web3.eth.get_balance(address)
                    balance_eth = float(self.web3.from_wei(balance_wei, 'ether'))
                    possible_sends = int(balance_eth / funding_reqs['cost_per_tx'])
                    
                    if possible_sends >= min_sends:
                        max_possible = min(possible_sends, max_sends)
                        funded_wallets.append((address, balance_eth, max_possible))
                    else:
                        eth_needed = funding_reqs['min_balance'] - balance_eth
                        pending_wallets.append((address, balance_eth, eth_needed))

                # Show funding status
                self.logger.info(f"\nWallet Funding Status:")
                self.logger.info(f"Funded wallets: {len(funded_wallets)}")
                self.logger.info(f"Pending wallets: {len(pending_wallets)}")

                if pending_wallets:
                    print("\nPending Wallets (Need Funding):")
                    pending_table = [[
                        addr, 
                        f"{bal:.6f}", 
                        f"{needed:.6f}",
                        int(bal / funding_reqs['cost_per_tx']),  # Current possible sends
                        min_sends - int(bal / funding_reqs['cost_per_tx'])  # Additional sends needed
                    ] for addr, bal, needed in pending_wallets]
                    print(tabulate(pending_table, 
                                 headers=['Address', 'Current Balance', 'ETH Needed', 'Current Sends', 'Additional Sends Needed'],
                                 tablefmt='grid'))

                if not funded_wallets:
                    self.logger.error("No funding wallets with sufficient balance found")
                    return

                # Get receiving wallets
                receiving_wallets = conn.execute('''
                    SELECT address 
                    FROM wallets 
                    WHERE wallet_type = 'receiving'
                    ORDER BY RANDOM()
                ''').fetchall()

                # Create distribution plan with funded wallets
                distribution_plan = []
                remaining_receivers = receiving_wallets.copy()
                
                for funding_address, balance, max_possible in funded_wallets:
                    num_sends = random.randint(min_sends, min(max_sends, max_possible))
                    
                    if not remaining_receivers:
                        break
                        
                    receivers = remaining_receivers[:num_sends]
                    remaining_receivers = remaining_receivers[num_sends:]
                    
                    for receiving_address in receivers:
                        distribution_plan.append((funding_address, receiving_address[0], eth_amount))

                # Save pending wallets status
                conn.execute('CREATE TABLE IF NOT EXISTS pending_funding_wallets (address TEXT PRIMARY KEY, current_balance TEXT, eth_needed TEXT, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)')
                
                # Clear previous pending entries
                conn.execute('DELETE FROM pending_funding_wallets')
                
                # Insert new pending wallets
                conn.executemany('''
                    INSERT INTO pending_funding_wallets (address, current_balance, eth_needed)
                    VALUES (?, ?, ?)
                ''', [(addr, str(bal), str(needed)) for addr, bal, needed in pending_wallets])

                # Insert distribution plan into database
                plan_id = self.insert_distribution_plan(conn, distribution_plan)
                
                # Show distribution summary
                self.logger.info("\nDistribution Plan Created:")
                self.logger.info(f"Plan ID: {plan_id}")
                self.logger.info(f"Total transactions: {len(distribution_plan)}")
                self.logger.info(f"Funded wallets used: {len(set(w[0] for w in distribution_plan))}")
                self.logger.info(f"Receiving wallets covered: {len(set(w[1] for w in distribution_plan))}")
                self.logger.info(f"Total ETH to be sent: {len(distribution_plan) * eth_amount:.6f}")
                
                if remaining_receivers:
                    self.logger.warning(f"\nWarning: {len(remaining_receivers)} receiving wallets not covered")
                    self.logger.warning("Consider funding pending wallets or increasing sends per wallet")

                conn.commit()

            except Exception as e:
                self.logger.error(f"Error creating distribution plan: {str(e)}")
                raise

    def send_transaction(self, private_key: str, to_address: str, amount_eth: float) -> Optional[str]:
        """
        Send ETH transaction and return transaction hash if successful.
        """
        try:
            account = Account.from_key(private_key)
            from_address = account.address
            
            # Get latest nonce from network
            nonce = self.web3.eth.get_transaction_count(from_address, 'latest')
            
            # Get gas estimate first
            gas_estimate = self.web3.eth.estimate_gas({
                'from': from_address,
                'to': to_address,
                'value': self.web3.to_wei(amount_eth, 'ether'),
                'nonce': nonce,
                'chainId': CHAIN_ID
            })

            transaction: TxParams = {
                'nonce': nonce,
                'to': to_address,
                'value': self.web3.to_wei(amount_eth, 'ether'),
                'gas': gas_estimate,
                'gasPrice': DEFAULT_GAS_PRICE,
                'chainId': CHAIN_ID
            }

            signed_txn = self.web3.eth.account.sign_transaction(transaction, private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            return self.web3.to_hex(tx_hash)

        except Exception as e:
            # Log the failed transaction
            self.tx_logger.log_failed_tx(
                from_addr=account.address,
                to_addr=to_address,
                amount=amount_eth,
                nonce=nonce,
                error=str(e),
                gas_used=gas_estimate,
                gas_price=DEFAULT_GAS_PRICE
            )
            return None

    def execute_distribution(self):
        """
        Execute pending distribution tasks and update wallet balances.
        Implements delays and batching for mainnet safety.
        """
        with sqlite3.connect(DB_PATH) as conn:
            try:
                # Get all pending tasks without filtering by plan_id
                tasks = conn.execute('''
                    SELECT dt.id, dt.receiving_wallet, dt.amount_eth, w.private_key
                    FROM distribution_tasks dt
                    JOIN wallets w ON dt.funding_wallet = w.address
                    WHERE dt.status = 'pending'
                    ORDER BY dt.id
                ''').fetchall()

                if not tasks:
                    self.logger.info("No pending distribution tasks found")
                    return

                self.logger.info(f"Found {len(tasks)} pending tasks to process")

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
                for task_id, receiving_wallet, amount_eth, private_key in progress_bar:
                    # Update progress description for current transaction
                    progress_bar.set_description(
                        f"To: {receiving_wallet}"
                    )
                    self.logger.debug(f"Processing transaction to {receiving_wallet} with amount {amount_eth} ETH")

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
                            WHERE id = ?
                        ''', (tx_hash, task_id))

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
                            WHERE id = ?
                        ''', (task_id,))
                        fail_count += 1

                    conn.commit()

                self.logger.info(f"Execution completed: {success_count} successful, {fail_count} failed")
                self.show_distribution_status()

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

    def resume_distribution(self):
        """Resume distribution by processing all pending transactions from funded wallets"""
        with sqlite3.connect(DB_PATH) as conn:
            try:
                # Calculate gas costs for balance check
                gas_cost_wei = DEFAULT_GAS_PRICE * GAS_LIMIT
                gas_cost_eth = float(self.web3.from_wei(gas_cost_wei, 'ether'))

                # Get all pending tasks with their funding wallet balances
                tasks = conn.execute('''
                    SELECT dt.id, dt.funding_wallet, dt.receiving_wallet, 
                           dt.amount_eth, w.private_key
                    FROM distribution_tasks dt
                    JOIN wallets w ON dt.funding_wallet = w.address
                    WHERE dt.status = 'pending'
                    ORDER BY dt.funding_wallet, dt.id
                ''').fetchall()

                if not tasks:
                    self.logger.info("No pending tasks found to resume")
                    return

                self.logger.info(f"Found {len(tasks)} pending tasks")

                # Group tasks by funding wallet
                wallet_tasks = {}
                for task in tasks:
                    if task[1] not in wallet_tasks:  # task[1] is funding_wallet
                        wallet_tasks[task[1]] = []
                    wallet_tasks[task[1]].append(task)

                # Check each funding wallet's balance
                funded_tasks = []
                pending_tasks = []

                for funding_wallet, wallet_task_group in wallet_tasks.items():
                    balance_wei = self.web3.eth.get_balance(funding_wallet)
                    balance_eth = float(self.web3.from_wei(balance_wei, 'ether'))
                    
                    total_needed = sum(task[3] for task in wallet_task_group)  # task[3] is amount_eth
                    total_needed_with_gas = total_needed + (gas_cost_eth * len(wallet_task_group))

                    if balance_eth >= total_needed_with_gas:
                        funded_tasks.extend(wallet_task_group)
                    else:
                        pending_tasks.extend(wallet_task_group)
                        # Update pending_funding_wallets table
                        eth_needed = total_needed_with_gas - balance_eth
                        conn.execute('''
                            INSERT OR REPLACE INTO pending_funding_wallets 
                            (address, current_balance, eth_needed, updated_at)
                            VALUES (?, ?, ?, datetime('now'))
                        ''', (funding_wallet, str(balance_eth), str(eth_needed)))

                if pending_tasks:
                    self.logger.warning(f"{len(pending_tasks)} tasks from unfunded wallets moved to pending")
                    # Show pending wallets
                    pending_wallets = conn.execute('SELECT * FROM pending_funding_wallets').fetchall()
                    if pending_wallets:
                        print("\nPending Wallets (Need Funding):")
                        pending_table = [[addr, bal, needed] for addr, bal, needed, _ in pending_wallets]
                        print(tabulate(pending_table, 
                                     headers=['Address', 'Current Balance', 'ETH Needed'],
                                     tablefmt='grid'))

                if not funded_tasks:
                    self.logger.info("No funded tasks available to resume")
                    return

                self.logger.info(f"Resuming with {len(funded_tasks)} funded tasks")

                # Initialize progress bar
                progress_bar = tqdm(
                    funded_tasks,
                    desc="Processing transactions",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                )

                tx_delay = MAINNET_TX_DELAY if MAINNET_MODE else TX_DELAY
                success_count = 0
                fail_count = 0

                # Process funded transactions
                for task_id, funding_wallet, receiving_wallet, amount_eth, private_key in progress_bar:
                    progress_bar.set_description(f"From: {funding_wallet} To: {receiving_wallet}")

                    tx_hash = self.send_transaction(
                        private_key=private_key,
                        to_address=receiving_wallet,
                        amount_eth=float(amount_eth)
                    )

                    if tx_hash:
                        conn.execute('''
                            UPDATE distribution_tasks 
                            SET status = 'sent',
                                tx_hash = ?,
                                executed_at = datetime('now')
                            WHERE id = ?
                        ''', (tx_hash, task_id))
                        success_count += 1
                        time.sleep(tx_delay)
                    else:
                        conn.execute('''
                            UPDATE distribution_tasks 
                            SET status = 'failed',
                                executed_at = datetime('now')
                            WHERE id = ?
                        ''', (task_id,))
                        fail_count += 1

                    conn.commit()

                self.logger.info(f"\nResume completed:")
                self.logger.info(f"Success: {success_count}")
                self.logger.info(f"Failed: {fail_count}")
                self.logger.info(f"Pending (unfunded): {len(pending_tasks)}")
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

    def process_pending_wallets(self):
        """Create a new distribution plan for previously pending wallets that are now funded"""
        with sqlite3.connect(DB_PATH) as conn:
            try:
                # Get pending wallets
                pending_wallets = conn.execute('''
                    SELECT address, current_balance, eth_needed 
                    FROM pending_funding_wallets
                    ORDER BY address
                ''').fetchall()

                if not pending_wallets:
                    self.logger.info("No pending wallets found")
                    return

                self.logger.info(f"Found {len(pending_wallets)} previously pending wallets")
                
                # Check current balances and create new plan for funded ones
                newly_funded = []
                still_pending = []
                
                # Get wallet counts for distribution calculation
                funding_count = conn.execute('SELECT COUNT(*) FROM wallets WHERE wallet_type = "funding"').fetchone()[0]
                receiving_count = conn.execute('SELECT COUNT(*) FROM wallets WHERE wallet_type = "receiving"').fetchone()[0]
                
                # Calculate dynamic distribution
                min_sends, max_sends = self.calculate_wallet_distribution(receiving_count, funding_count)
                funding_reqs = self.calculate_funding_requirements(DEFAULT_ETH_AMOUNT, min_sends, max_sends)

                for address, _, _ in pending_wallets:
                    balance_wei = self.web3.eth.get_balance(address)
                    balance_eth = float(self.web3.from_wei(balance_wei, 'ether'))
                    possible_sends = int(balance_eth / funding_reqs['cost_per_tx'])
                    
                    if possible_sends >= min_sends:
                        max_possible = min(possible_sends, max_sends)
                        newly_funded.append((address, balance_eth, max_possible))
                    else:
                        eth_needed = funding_reqs['min_balance'] - balance_eth
                        still_pending.append((address, balance_eth, eth_needed))

                # Show status
                self.logger.info(f"\nWallet Status:")
                self.logger.info(f"Newly funded wallets: {len(newly_funded)}")
                self.logger.info(f"Still pending wallets: {len(still_pending)}")

                if still_pending:
                    print("\nStill Pending Wallets:")
                    pending_table = [[
                        addr, 
                        f"{bal:.6f}", 
                        f"{needed:.6f}",
                        int(bal / funding_reqs['cost_per_tx']),
                        min_sends - int(bal / funding_reqs['cost_per_tx'])
                    ] for addr, bal, needed in still_pending]
                    print(tabulate(pending_table, 
                                 headers=['Address', 'Current Balance', 'ETH Needed', 'Current Sends', 'Additional Sends Needed'],
                                 tablefmt='grid'))

                if not newly_funded:
                    self.logger.info("No newly funded wallets to process")
                    return

                # Create new distribution plan for newly funded wallets
                self.logger.info("\nCreating new distribution plan for funded wallets...")
                
                # Update pending_funding_wallets table
                conn.execute('DELETE FROM pending_funding_wallets')
                if still_pending:
                    conn.executemany('''
                        INSERT INTO pending_funding_wallets (address, current_balance, eth_needed)
                        VALUES (?, ?, ?)
                    ''', [(addr, str(bal), str(needed)) for addr, bal, needed in still_pending])

                # Create new distribution plan
                return self.create_distribution_plan()

            except Exception as e:
                self.logger.error(f"Error processing pending wallets: {str(e)}")
                raise

    def insert_distribution_plan(self, conn, distribution_plan):
        """Insert distribution plan into database and return plan ID"""
        try:
            # Create a new plan
            cursor = conn.execute('''
                INSERT INTO distribution_plan (created_at, status)
                VALUES (datetime('now'), 'pending')
            ''')
            plan_id = cursor.lastrowid

            # Insert tasks
            conn.executemany('''
                INSERT INTO distribution_tasks 
                (plan_id, funding_wallet, receiving_wallet, amount_eth, status)
                VALUES (?, ?, ?, ?, 'pending')
            ''', [(plan_id, fw, rw, amount) for fw, rw, amount in distribution_plan])

            conn.commit()
            return plan_id

        except Exception as e:
            self.logger.error(f"Error inserting distribution plan: {str(e)}")
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
    parser.add_argument('--process-pending', action='store_true', 
                       help='Process previously pending wallets that are now funded')
    
    args = parser.parse_args()
    distributor = Distributor()
    
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
        distributor.check_all_receiving_balances()
    elif args.check_funding:
        distributor.check_all_funding_balances()
    elif args.update_amount:
        distributor.update_distribution_amount(args.update_amount)
    elif args.resend_all:
        distributor.resend_to_all(args.resend_all)
    elif args.resume or args.resume_from:
        distributor.resume_distribution()
    elif args.check_funding_needed:
        distributor.show_underfunded_wallets(eth_amount=args.check_amount)
    elif args.process_pending:
        distributor.process_pending_wallets()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()