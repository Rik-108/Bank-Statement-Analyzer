import logging
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import io
from watsonx_promptlab import generate_financial_analysis, detect_fraud, assess_loan_eligibility

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Serve static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r") as f:
        return f.read()

async def process_statement(file: UploadFile):
    """Helper function to process the uploaded CSV file and generate a summary."""
    try:
        logger.info("Received file upload: %s", file.filename)

        # Validate file type
        if not file.filename.endswith('.csv'):
            logger.error("Invalid file type: %s", file.filename)
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        # Read CSV file
        contents = await file.read()
        try:
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            logger.info("Successfully read CSV file with %d rows", len(df))
        except Exception as e:
            logger.error("Failed to parse CSV file: %s", str(e))
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

        # Validate required columns
        required_columns = ['CREDIT', 'DEBIT', 'BALANCE']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error("Missing required columns in CSV: %s", missing_columns)
            raise HTTPException(status_code=400, detail=f"Missing required columns in CSV: {missing_columns}")

        # Process the bank statement
        try:
            total_credited = df['CREDIT'].replace(r'^\+$', '0', regex=True).astype(float).sum()
            total_debited = df['DEBIT'].replace(r'^-+$', '0', regex=True).astype(float).sum()
            average_balance = df['BALANCE'].astype(float).mean()
            closing_balance = df['BALANCE'].iloc[-1]

            # Identify red flags
            days_below_500 = (df['BALANCE'].astype(float) < 500).sum()
            high_value_debits = (df['DEBIT'].replace(r'^-+$', '0', regex=True).astype(float) < -1000).sum()
            frequent_small_debits = (df['DEBIT'].replace(r'^-+$', '0', regex=True).astype(float).between(-200, 0)).sum()

            # Additional metrics for loan eligibility
            # Debt-to-Income Ratio (approximated as total debits / total credits)
            dti_ratio = (total_debited / total_credited * 100) if total_credited > 0 else float('inf')

            # Savings Rate: (Total Credits - Total Debits) / Total Credits
            savings_rate = ((total_credited - total_debited) / total_credited * 100) if total_credited > 0 else 0

            # Maximum Balance Drop: Largest single drop in balance
            balance_drops = df['BALANCE'].astype(float).diff().dropna()
            max_balance_drop = balance_drops.min() if not balance_drops.empty else 0

            # Number of Overdrafts: Count of negative balance occurrences
            overdrafts = (df['BALANCE'].astype(float) < 0).sum()

        except Exception as e:
            logger.error("Error processing bank statement data: %s", str(e))
            raise HTTPException(status_code=500, detail=f"Error processing bank statement data: {str(e)}")

        # Generate summary
        summary = (
            f"📄 Bank Statement Summary:\n"
            f"• Total credited: ₹{total_credited:,.2f}\n"
            f"• Total debited: ₹{total_debited:,.2f}\n"
            f"• Average balance: ₹{average_balance:,.2f}\n"
            f"• Closing balance: ₹{closing_balance:,.2f}\n\n"
            f"🚩 Red Flags:\n"
            f"• {days_below_500} days had balance below ₹500\n"
            f"• {high_value_debits} high-value debits (over ₹1000)\n"
            f"• {'✔️' if frequent_small_debits > 50 else '❌'} Frequent small debits (< ₹200) detected: {frequent_small_debits}\n\n"
            f"💰 Loan Eligibility Metrics:\n"
            f"• Debt-to-Income Ratio: {dti_ratio:.2f}% {'(High)' if dti_ratio > 40 else '(Acceptable)'}\n"
            f"• Savings Rate: {savings_rate:.2f}% {'(Good)' if savings_rate > 20 else '(Low)'}\n"
            f"• Maximum Balance Drop: ₹{max_balance_drop:,.2f}\n"
            f"• Overdrafts: {overdrafts} occurrences"
        )
        logger.info("Generated summary: %s", summary)

        # Generate financial analysis
        logger.info("Calling Watsonx Prompt Lab for financial analysis")
        financial_insight = generate_financial_analysis(summary)
        logger.info("Received financial analysis from Watsonx Prompt Lab: %s", financial_insight)

        return summary, financial_insight, None
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("Error processing file: %s", str(e))
        return None, None, f"Error processing file: {str(e)}"

@app.post("/analyze/")
async def analyze_endpoint(file: UploadFile = File(...)):
    """Endpoint to analyze the bank statement and return a summary and financial analysis."""
    summary, financial_insight, error = await process_statement(file)
    if error:
        raise HTTPException(status_code=500, detail=error)

    # Use the financial_insight as promptlab_insight (no validation)
    promptlab_insight = financial_insight
    logger.info("Using financial_insight as promptlab_insight for /analyze/ endpoint: %s", promptlab_insight)

    return {
        "summary": summary,
        "promptlab_insight": promptlab_insight
    }

@app.post("/detect_fraud/")
async def detect_fraud_endpoint(file: UploadFile = File(...)):
    """Endpoint to detect potential fraud in the bank statement."""
    summary, financial_insight, error = await process_statement(file)
    if error:
        raise HTTPException(status_code=500, detail=error)

    # Call Watsonx Prompt Lab for fraud detection
    logger.info("Calling Watsonx Prompt Lab for fraud detection")
    fraud_insight = detect_fraud(summary)
    logger.info("Received fraud detection insight from Watsonx Prompt Lab: %s", fraud_insight)

    return {
        "summary": summary,
        "financial_analysis": financial_insight,
        "fraud_insight": fraud_insight
    }

@app.post("/assess_loan_eligibility/")
async def assess_loan_eligibility_endpoint(file: UploadFile = File(...)):
    """Endpoint to assess loan eligibility based on the bank statement."""
    summary, financial_insight, error = await process_statement(file)
    if error:
        raise HTTPException(status_code=500, detail=error)

    # Call Watsonx Prompt Lab for loan eligibility assessment
    logger.info("Calling Watsonx Prompt Lab for loan eligibility assessment")
    loan_eligibility_insight = assess_loan_eligibility(summary)
    logger.info("Received loan eligibility insight from Watsonx Prompt Lab: %s", loan_eligibility_insight)

    return {
        "summary": summary,
        "financial_analysis": financial_insight,
        "loan_eligibility_insight": loan_eligibility_insight
    }