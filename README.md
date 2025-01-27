# Fundoor Distribution System

A comprehensive system for managing ETH distribution and NFT minting across multiple wallets.

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.dist .env
```
Edit `.env` with your specific configuration:
- `RPC_ENDPOINT`: Your blockchain RPC endpoint
- `CHAIN_ID`: Network chain ID
- `NFT_CONTRACT_ADDRESS`: Address of the NFT contract
- `DEFAULT_ETH_AMOUNT`: Default amount of ETH to distribute
- Other parameters as needed

4. Prepare wallet files:
- Create `funding.csv` with funding wallet addresses and private keys
- Create `receiving.csv` with receiving wallet addresses and private keys
Format for both files:
```csv
address,private_key
0x123...,abc...
```

## ETH Distribution Features

### Import Wallets
```bash
python dist.py --import-wallets
```

### Check Balances
```bash
# Check specific wallet
python dist.py --check-balance 0x123...

# Check all receiving wallets
python dist.py --check-receiving

# Check all funding wallets
python dist.py --check-funding
```

### Create and Execute Distribution
```bash
# Create distribution plan
python dist.py --create-plan --amount 0.0009

# Execute distribution
python dist.py --execute

# Show distribution status
python dist.py --status
```

### Control Distribution
```bash
# Update amount for pending transactions
python dist.py --update-amount 0.001

# Reset and resend all with new amount
python dist.py --resend-all 0.001

# Resume from last successful transaction
python dist.py --resume

# Resume from specific wallet
python dist.py --resume-from 0x123...
```

## NFT Minting Features

### Start Minting
```bash
# Start minting with default token ID (1) and amount (1)
python minter.py --mint

# Mint with custom token ID and amount
python minter.py --mint --token-id 2 --amount 1
```

### Control Minting
```bash
# Check minting status
python minter.py --status

# Stop minting process
python minter.py --stop

# Resume minting
python minter.py --resume --token-id 1 --amount 1

# Restart minting (resets all statuses)
python minter.py --restart --token-id 1 --amount 1
```

## Database Management

The system uses SQLite for data persistence. Database files:
- `distribution.db`: Main database file
- `distribution.log`: Distribution process logs
- `minting.log`: NFT minting logs

To reset the database for testing:
1. Stop any running processes
2. Delete the database files:
```bash
rm distribution.db*
```
3. Restart the process to recreate the database

## Safety Features

- Mainnet protection with configurable delays
- Batch processing with pauses
- Transaction retry mechanism
- Comprehensive logging
- Progress tracking
- Error handling and recovery

## Environment Variables

Key configuration in `.env`:
```
# Network
RPC_ENDPOINT="https://api.testnet.abs.xyz"
CHAIN_ID=11124

# Contract
NFT_CONTRACT_ADDRESS="0x..."

# Distribution
DEFAULT_ETH_AMOUNT=0.0009

# Safety
MAINNET_MODE=false
TX_DELAY=1
BATCH_SIZE=50
```

## Error Recovery

If processes are interrupted:
1. Check logs for last successful operation
2. Use status commands to verify state
3. Resume operations using appropriate resume command
4. For complete reset, use restart commands

## Development

- Python 3.8+ required
- Uses Web3.py for blockchain interaction
- SQLite for local state management
- Environment-based configuration
- Comprehensive logging and error handling
