import os
import logging
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.credentials import Credentials

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

def _setup_model():
    """Initialize the Watsonx API model and credentials."""
    # Verify environment variables
    region = os.getenv('WATSONX_REGION')
    api_key = os.getenv('WATSONX_API_KEY')
    project_id = os.getenv('WATSONX_PROJECT_ID')

    logger.debug(f"Environment variables - Region: {region}, Project ID: {project_id}, API Key: {'Set' if api_key else 'Not Set'}")

    if not all([region, api_key, project_id]):
        logger.error("Missing required environment variables for Watsonx API")
        return None, "Error: Missing required environment variables for Watsonx API"

    creds = Credentials(
        url=f"https://{region}.ml.cloud.ibm.com",
        api_key=api_key
    )

    try:
        logger.debug("Initializing ModelInference with credentials")
        model = ModelInference(
            model_id="meta-llama/llama-3-3-70b-instruct",
            credentials=creds,
            project_id=project_id
        )
        return model, None
    except Exception as e:
        logger.error(f"Failed to initialize ModelInference: {str(e)}")
        return None, f"Error: Failed to initialize ModelInference: {str(e)}"

def _handle_response(response, prompt, summary_text):
    """Handle the response from Watsonx API and extract the generated text."""
    logger.debug(f"Raw response type: {type(response)}")
    logger.debug(f"Raw response: {response}")

    if isinstance(response, str):
        logger.info("Response is a string")
        if prompt in response or summary_text in response:
            logger.warning("Response contains prompt or summary; attempting to extract analysis")
            analysis_start = response.find("**Analysis:**") + len("**Analysis:**")
            if analysis_start == -1:
                logger.error("Analysis marker not found in response")
                return "Error: Model failed to generate valid analysis"
            cleaned_response = response[analysis_start:].strip()
            if not cleaned_response or any(seq in cleaned_response for seq in ["**Instructions**", "**Bank Statement Summary**"]):
                logger.error("No valid analysis found in response")
                return "Error: Model failed to generate valid analysis"
            return cleaned_response
        return response.strip()
    elif isinstance(response, dict):
        logger.info("Response is a dictionary")
        if 'results' not in response:
            logger.error(f"Missing 'results' key in response: {response}")
            return "Error: Missing 'results' key in response"
        if not isinstance(response['results'], list):
            logger.error(f"'results' is not a list: {type(response['results'])}")
            return f"Error: 'results' is not a list, got {type(response['results'])}"
        if not response['results']:
            logger.error("Empty 'results' list in response")
            return "Error: Empty 'results' list in response"
        generated_text = response['results'][0].get('generated_text', '')
        if not generated_text:
            logger.error("Generated text is empty in response")
            return "Error: Model failed to generate analysis"
        return generated_text.strip()
    elif isinstance(response, int):
        logger.error(f"Received integer response: {response}, likely a status code or token count")
        return "Error: Watsonx API returned an unexpected integer response"
    else:
        logger.error(f"Unsupported response type: {type(response)}")
        return "Error: Unsupported response type"

def generate_financial_analysis(summary_text):
    """Generate a comprehensive financial analysis of the bank statement summary using Watsonx API."""
    model, error = _setup_model()
    if error:
        return error

    prompt = f"""

{summary_text}

**Analysis:**
"""

    try:
        logger.debug("Sending financial analysis prompt to Watsonx API")
        logger.debug(f"Prompt length: {len(prompt)} characters")

        response = model.generate_text(
            prompt,
            params={
                GenParams.MAX_NEW_TOKENS: 8192,
                GenParams.TEMPERATURE: 0.7,
                GenParams.DECODING_METHOD: "sample",
                GenParams.STOP_SEQUENCES: ["**Instructions**", "**Bank Statement Summary**", "**Analysis:**", "\n\n"]
            }
        )
        logger.debug("Successfully received response from Watsonx API")
        return _handle_response(response, prompt, summary_text)

    except Exception as e:
        logger.error(f"Watsonx API call for financial analysis failed: {str(e)}")
        return f"Error: Watsonx API call failed: {str(e)}"

def detect_fraud(summary_text):
    """Analyze the bank statement summary for potential fraud using Watsonx API."""
    model, error = _setup_model()
    if error:
        return error

    prompt = f"""
**Instructions**
Provide a fraud detection analysis of the bank statement summary.
**Bank Statement Summary**
{summary_text}

**Analysis:**
"""

    try:
        logger.debug("Sending fraud detection prompt to Watsonx API")
        logger.debug(f"Prompt length: {len(prompt)} characters")

        response = model.generate_text(
            prompt,
            params={
                GenParams.MAX_NEW_TOKENS: 8192,
                GenParams.TEMPERATURE: 0.7,
                GenParams.DECODING_METHOD: "sample",
                GenParams.STOP_SEQUENCES: ["**Instructions**", "**Bank Statement Summary**", "**Analysis:**", "\n\n"]
            }
        )
        logger.debug("Successfully received response from Watsonx API")
        return _handle_response(response, prompt, summary_text)

    except Exception as e:
        logger.error(f"Watsonx API call for fraud detection failed: {str(e)}")
        return f"Error: Watsonx API call failed: {str(e)}"

def assess_loan_eligibility(summary_text):
    """Assess the account holder's loan eligibility based on the bank statement summary using Watsonx API."""
    model, error = _setup_model()
    if error:
        return error

    prompt = f"""
**Instructions**
Check the Red Flags for loan eligibility. give the answer in Yes it is eligible for loan or No it is not eligible for loan.
**Bank Statement Summary**
{summary_text}

**Analysis:**
"""

    try:
        logger.debug("Sending loan eligibility prompt to Watsonx API")
        logger.debug(f"Prompt length: {len(prompt)} characters")

        response = model.generate_text(
            prompt,
            params={
                GenParams.MAX_NEW_TOKENS: 8192,
                GenParams.TEMPERATURE: 0.7,
                GenParams.DECODING_METHOD: "sample",
                GenParams.STOP_SEQUENCES: ["**Instructions**", "**Bank Statement Summary**", "**Analysis:**", "\n\n"]
            }
        )
        logger.debug("Successfully received response from Watsonx API")
        return _handle_response(response, prompt, summary_text)

    except Exception as e:
        logger.error(f"Watsonx API call for loan eligibility failed: {str(e)}")
        return f"Error: Watsonx API call failed: {str(e)}"