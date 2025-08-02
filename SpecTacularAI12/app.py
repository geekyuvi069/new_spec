import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, render_template_string, redirect, url_for, flash
from flask_cors import CORS
from werkzeug.utils import secure_filename

from src.preprocessing import extract_text_from_pdf, clean_text, split_into_chunks
from src.semantic_search import build_index, search
from src.validation_engine import ValidationEngine
from src.mapping_engine import MappingEngine
from src.traceability_matrix import TraceabilityMatrix
from src.pdf_generator import PDFGenerator

# Initialize Flask
app = Flask(__name__, static_folder="web/static", template_folder="web")
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
CORS(app)

# Global variables
document_chunks = []
srs_requirements = []
test_cases = []
validation_engine = ValidationEngine()
mapping_engine = MappingEngine()
traceability_matrix = TraceabilityMatrix()
pdf_generator = PDFGenerator()

@app.route("/")
def index():
    return send_from_directory("web", "index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global document_chunks, srs_requirements
    
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Create date-based folder
    date_folder = datetime.now().strftime("%Y-%m-%d")
    save_folder = os.path.join("data", date_folder)
    os.makedirs(save_folder, exist_ok=True)

    # Save uploaded file
    filename = secure_filename(file.filename)
    save_path = os.path.join(save_folder, filename)
    file.save(save_path)

    try:
        print(f"Starting processing of {filename}...")
        
        # Process document with optimizations
        raw_text = extract_text_from_pdf(save_path)
        print(f"Extracted {len(raw_text)} characters")
        
        clean = clean_text(raw_text)
        print(f"Cleaned text: {len(clean)} characters")
        
        # Use larger chunks for faster processing
        chunks = split_into_chunks(clean, chunk_size=300)
        document_chunks = chunks
        print(f"Created {len(chunks)} chunks")
        
        # Extract requirements with enhanced method
        srs_requirements = mapping_engine.extract_requirements(chunks)
        print(f"Extracted {len(srs_requirements)} requirements")

        # Build embeddings index (optimized)
        if chunks:  # Only build if we have chunks
            build_index(chunks)
            print("Built semantic index")

        return jsonify({
            "message": "Document processed successfully",
            "chunks": len(chunks),
            "requirements": len(srs_requirements),
            "processing_time": "optimized",
            "savedPath": save_path
        })
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route("/upload_testcases", methods=["POST"])
def upload_testcases():
    """Upload existing test cases file for comparison"""
    global test_cases
    
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Read JSON test cases
        content = file.read().decode('utf-8')
        existing_test_cases = json.loads(content)
        
        # Ensure existing_test_cases is a list
        if not isinstance(existing_test_cases, list):
            return jsonify({"error": "Test cases file must contain a JSON array"}), 400
        
        # Merge with current test cases (avoid duplicates by ID)
        existing_ids = {tc.get('id') if isinstance(tc, dict) else None for tc in test_cases}
        for tc in existing_test_cases:
            if isinstance(tc, dict):
                tc_id = tc.get('id')
                if tc_id not in existing_ids:
                    test_cases.append(tc)
            else:
                # Handle case where tc is not a dict - convert or skip
                print(f"Warning: Skipping non-dict test case: {tc}")
        
        return jsonify({
            "message": "Test cases uploaded successfully",
            "total_test_cases": len(test_cases)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process test cases: {str(e)}"}), 500

@app.route("/query", methods=["POST"])
def query():
    global test_cases
    
    data = request.get_json()
    query_text = data.get("query", "").strip()
    if not query_text:
        return jsonify({"error": "Empty query."}), 400

    try:
        # Retrieve top similar chunks
        retrieved_chunks = search(query_text, top_k=3)

        # Generate test cases from retrieved text
        generated_test_cases = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            test_case = {
                "id": f"TC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                "title": f"Test Case for: {query_text[:50]}...",
                "description": f"Generated test case based on query: {query_text}",
                "steps": _generate_test_steps(chunk, query_text),
                "expected": "System should behave as specified in the requirements",
                "priority": "Medium",
                "type": "Functional",
                "requirement_id": _extract_requirement_id(chunk),
                "status": "Generated",
                "query": query_text,
                "chunk_source": chunk[:200] + "..." if len(chunk) > 200 else chunk
            }
            generated_test_cases.append(test_case)

        # Add to global test cases list
        test_cases.extend(generated_test_cases)

        return jsonify({
            "query": query_text,
            "testCases": generated_test_cases
        })
    except Exception as e:
        return jsonify({"error": f"Query processing failed: {str(e)}"}), 500

@app.route("/validate", methods=["POST"])
def validate_test_cases():
    """Validate all test cases"""
    try:
        # Clean up test cases first
        valid_test_cases = _clean_test_cases()
        
        validation_results = validation_engine.validate_test_cases(valid_test_cases, srs_requirements)
        
        return jsonify({
            "validation_results": validation_results,
            "summary": {
                "total": len(valid_test_cases),
                "valid": sum(1 for r in validation_results if r.get('is_valid', False)),
                "invalid": sum(1 for r in validation_results if not r.get('is_valid', True))
            }
        })
    except Exception as e:
        return jsonify({"error": f"Validation failed: {str(e)}"}), 500

@app.route("/map", methods=["POST"])
def map_test_cases():
    """Map test cases to requirements"""
    try:
        # Filter out any non-dict test cases
        valid_test_cases = [tc for tc in test_cases if isinstance(tc, dict)]
        
        if len(valid_test_cases) != len(test_cases):
            print(f"Warning: Found {len(test_cases) - len(valid_test_cases)} non-dict test cases during mapping")
        
        mapping_results = mapping_engine.map_test_cases_to_requirements(valid_test_cases, srs_requirements)
        
        return jsonify({
            "mapping_results": mapping_results,
            "summary": {
                "total_test_cases": len(valid_test_cases),
                "mapped_test_cases": len([m for m in mapping_results if m['mapped_requirements']]),
                "total_requirements": len(srs_requirements),
                "covered_requirements": len(set([
                    req['requirement_id'] 
                    for m in mapping_results 
                    for req in m.get('mapped_requirements', [])
                    if isinstance(req, dict) and 'requirement_id' in req
                ]))
            }
        })
    except Exception as e:
        return jsonify({"error": f"Mapping failed: {str(e)}"}), 500

@app.route("/traceability", methods=["POST"])
def generate_traceability_matrix():
    """Generate traceability matrix"""
    try:
        # Filter out any non-dict test cases
        valid_test_cases = [tc for tc in test_cases if isinstance(tc, dict)]
        
        if len(valid_test_cases) != len(test_cases):
            print(f"Warning: Found {len(test_cases) - len(valid_test_cases)} non-dict test cases during traceability")
        
        matrix_data = traceability_matrix.generate_matrix(srs_requirements, valid_test_cases)
        
        return jsonify({
            "matrix": matrix_data,
            "coverage_stats": traceability_matrix.calculate_coverage_stats(matrix_data)
        })
    except Exception as e:
        return jsonify({"error": f"Traceability matrix generation failed: {str(e)}"}), 500

@app.route("/export/pdf", methods=["POST"])
def export_pdf():
    """Export test cases and traceability matrix to PDF"""
    try:
        data = request.get_json()
        export_type = data.get("type", "test_cases")
        
        if export_type == "test_cases":
            pdf_path = pdf_generator.generate_test_cases_pdf(test_cases)
        elif export_type == "traceability":
            matrix_data = traceability_matrix.generate_matrix(srs_requirements, test_cases)
            pdf_path = pdf_generator.generate_traceability_pdf(matrix_data)
        elif export_type == "validation":
            validation_results = validation_engine.validate_test_cases(test_cases, srs_requirements)
            pdf_path = pdf_generator.generate_validation_pdf(validation_results)
        else:
            return jsonify({"error": "Invalid export type"}), 400
        
        return jsonify({
            "message": "PDF generated successfully",
            "download_url": f"/download/{os.path.basename(pdf_path)}"
        })
    except Exception as e:
        return jsonify({"error": f"PDF export failed: {str(e)}"}), 500

@app.route("/export/excel", methods=["POST"])
def export_excel():
    """Export traceability matrix to Excel"""
    try:
        matrix_data = traceability_matrix.generate_matrix(srs_requirements, test_cases)
        excel_path = traceability_matrix.export_to_excel(matrix_data)
        
        return jsonify({
            "message": "Excel file generated successfully",
            "download_url": f"/download/{os.path.basename(excel_path)}"
        })
    except Exception as e:
        return jsonify({"error": f"Excel export failed: {str(e)}"}), 500

@app.route("/download/<filename>")
def download_file(filename):
    """Download generated files"""
    return send_from_directory("data/exports", filename, as_attachment=True)

def _generate_test_steps(chunk, query):
    """Generate test steps based on chunk content and query"""
    # Enhanced rule-based test step generation
    steps = []
    query_lower = query.lower()
    chunk_lower = chunk.lower()
    
    # User Registration scenarios
    if any(keyword in query_lower for keyword in ["registration", "register", "signup", "sign up", "create account"]):
        steps = [
            "1. Navigate to user registration page",
            "2. Enter valid user information (name, email, password)",
            "3. Click Register button",
            "4. Check for registration success message",
            "5. Check email inbox for verification email",
            "6. Click verification link in email",
            "7. Verify account activation"
        ]
    
    # Login scenarios
    elif any(keyword in query_lower for keyword in ["login", "sign in", "authentication", "signin"]):
        steps = [
            "1. Navigate to login page",
            "2. Enter valid username/email",
            "3. Enter correct password",
            "4. Click login button",
            "5. Verify redirect to dashboard or homepage",
            "6. Check for user session establishment"
        ]
    
    # Password Reset scenarios
    elif any(keyword in query_lower for keyword in ["password reset", "forgot password", "reset password"]):
        steps = [
            "1. Navigate to login page",
            "2. Click 'Forgot Password' link",
            "3. Enter registered email address",
            "4. Submit password reset request",
            "5. Check email for reset link",
            "6. Click reset link within 24 hours",
            "7. Enter new password",
            "8. Confirm password change",
            "9. Login with new password"
        ]
    
    # Shopping Cart scenarios
    elif any(keyword in query_lower for keyword in ["cart", "shopping cart", "add to cart"]):
        steps = [
            "1. Browse product catalog",
            "2. Select desired product",
            "3. Choose quantity and options",
            "4. Click 'Add to Cart' button",
            "5. Verify item appears in cart",
            "6. Check quantity and price accuracy",
            "7. Test cart persistence across sessions"
        ]
    
    # Checkout/Payment scenarios
    elif any(keyword in query_lower for keyword in ["checkout", "payment", "purchase", "buy"]):
        steps = [
            "1. Add products to shopping cart",
            "2. Proceed to checkout",
            "3. Enter shipping information",
            "4. Select shipping method",
            "5. Choose payment method",
            "6. Enter payment details",
            "7. Review order summary",
            "8. Complete purchase",
            "9. Verify order confirmation"
        ]
    
    # Search functionality
    elif any(keyword in query_lower for keyword in ["search", "find", "filter"]):
        steps = [
            "1. Navigate to search interface",
            "2. Enter search keywords",
            "3. Apply relevant filters",
            "4. Execute search",
            "5. Review search results",
            "6. Verify result relevance",
            "7. Test result sorting options"
        ]
    
    # Product management
    elif any(keyword in query_lower for keyword in ["product", "inventory", "catalog"]):
        steps = [
            "1. Login as merchant/admin",
            "2. Navigate to product management",
            "3. Click 'Add New Product'",
            "4. Fill in product details",
            "5. Upload product images",
            "6. Set pricing and inventory",
            "7. Publish product",
            "8. Verify product appears in catalog"
        ]
    
    # Order management
    elif any(keyword in query_lower for keyword in ["order", "tracking", "fulfillment"]):
        steps = [
            "1. Access order management system",
            "2. Locate specific order",
            "3. Update order status",
            "4. Add tracking information",
            "5. Send customer notification",
            "6. Verify status update",
            "7. Check customer notification delivery"
        ]
    
    # Validation/Testing scenarios
    elif any(keyword in query_lower for keyword in ["validate", "validation", "verify", "test"]):
        steps = [
            "1. Prepare test data and environment",
            "2. Execute validation process",
            "3. Input valid test data",
            "4. Verify successful validation",
            "5. Input invalid test data",
            "6. Verify error handling",
            "7. Check error messages for clarity"
        ]
    
    # Security/Access Control
    elif any(keyword in query_lower for keyword in ["security", "access", "permission", "role"]):
        steps = [
            "1. Login with test user account",
            "2. Attempt to access restricted feature",
            "3. Verify access control enforcement",
            "4. Test with different user roles",
            "5. Verify appropriate permissions",
            "6. Check error messages for unauthorized access"
        ]
    
    # Performance testing
    elif any(keyword in query_lower for keyword in ["performance", "load", "speed", "response time"]):
        steps = [
            "1. Set up performance monitoring",
            "2. Execute performance test scenario",
            "3. Measure response times",
            "4. Monitor system resources",
            "5. Verify performance meets requirements",
            "6. Document performance metrics"
        ]
    
    # Mobile testing
    elif any(keyword in query_lower for keyword in ["mobile", "responsive", "device"]):
        steps = [
            "1. Access application on mobile device",
            "2. Test touch interface functionality",
            "3. Verify responsive design",
            "4. Test navigation on small screen",
            "5. Verify mobile-specific features",
            "6. Check performance on mobile network"
        ]
    
    # Generic fallback with more structure
    else:
        # Analyze chunk content for better context
        if any(keyword in chunk_lower for keyword in ["user", "customer", "account"]):
            steps.append("1. Login as test user")
            steps.append("2. Navigate to relevant feature")
        elif any(keyword in chunk_lower for keyword in ["admin", "management", "configure"]):
            steps.append("1. Login as administrator")
            steps.append("2. Access management interface")
        else:
            steps.append("1. Navigate to feature under test")
        
        if any(keyword in chunk_lower for keyword in ["input", "enter", "data"]):
            steps.append(f"{len(steps)+1}. Enter required test data")
        
        if any(keyword in chunk_lower for keyword in ["click", "submit", "save"]):
            steps.append(f"{len(steps)+1}. Execute primary action")
        
        if any(keyword in chunk_lower for keyword in ["verify", "check", "validate"]):
            steps.append(f"{len(steps)+1}. Verify expected results")
        else:
            steps.append(f"{len(steps)+1}. Confirm successful operation")
    
    return "\n".join(steps) if steps else "1. Execute test scenario\n2. Verify expected results"

def _extract_requirement_id(chunk):
    """Extract requirement ID from chunk if available"""
    # Simple pattern matching for requirement IDs
    import re
    patterns = [r'REQ[_-]?\d+', r'R\d+', r'FR[_-]?\d+', r'NFR[_-]?\d+']
    
    for pattern in patterns:
        match = re.search(pattern, chunk, re.IGNORECASE)
        if match:
            return match.group()
    
    return None

def _clean_test_cases():
    """Clean up the global test_cases list to ensure all items are valid dictionaries"""
    global test_cases
    original_count = len(test_cases)
    test_cases = [tc for tc in test_cases if isinstance(tc, dict) and tc.get('id')]
    
    if len(test_cases) != original_count:
        print(f"Cleaned test cases: removed {original_count - len(test_cases)} invalid items")
    
    return test_cases

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/exports", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)