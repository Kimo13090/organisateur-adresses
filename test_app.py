#!/usr/bin/env python3
"""
Test script for the address organizer application
"""
import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import (
    detect_columns_smart,
    validate_input_data,
    calculate_route_quality_metrics,
    find_center_city_point,
    filter_addresses_by_proximity,
    optimize_delivery_route_from_center
)

def test_column_detection():
    """Test the smart column detection function"""
    print("Testing column detection...")
    
    # Create test data
    test_data = pd.DataFrame({
        'client_adresse': ['123 rue de la Paix', '456 avenue des Champs'],
        'code_postal': ['75001', '75008'],
        'ville': ['Paris', 'Paris'],
        'other_col': ['data1', 'data2']
    })
    
    address_col, postal_col, city_col = detect_columns_smart(test_data)
    
    assert address_col == 'client_adresse', f"Expected 'client_adresse', got {address_col}"
    assert postal_col == 'code_postal', f"Expected 'code_postal', got {postal_col}"
    assert city_col == 'ville', f"Expected 'ville', got {city_col}"
    
    print("✅ Column detection test passed")

def test_data_validation():
    """Test the data validation function"""
    print("Testing data validation...")
    
    # Valid data
    valid_data = pd.DataFrame({
        'adresse': ['123 rue de la Paix', '456 avenue des Champs'],
        'cp': ['75001', '75008'],
        'ville': ['Paris', 'Paris']
    })
    
    is_valid, message = validate_input_data(valid_data, 'adresse', 'cp', 'ville')
    assert is_valid, f"Valid data should pass validation: {message}"
    
    # Invalid data - missing column
    invalid_data = pd.DataFrame({
        'adresse': ['123 rue de la Paix'],
        'ville': ['Paris']
    })
    
    is_valid, message = validate_input_data(invalid_data, 'adresse', 'cp', 'ville')
    assert not is_valid, "Invalid data should fail validation"
    
    print("✅ Data validation test passed")

def test_route_quality_metrics():
    """Test the route quality metrics calculation"""
    print("Testing route quality metrics...")
    
    # Create test route data
    test_route = pd.DataFrame({
        'lat': [48.8566, 48.8606, 48.8516],
        'lon': [2.3522, 2.3376, 2.3478],
        'adresse': ['Point A', 'Point B', 'Point C']
    })
    
    metrics = calculate_route_quality_metrics(test_route)
    
    assert 'total_distance' in metrics, "Should calculate total distance"
    assert 'quality_score' in metrics, "Should calculate quality score"
    assert metrics['total_distance'] > 0, "Total distance should be positive"
    
    print("✅ Route quality metrics test passed")

def test_center_point_detection():
    """Test the center point detection"""
    print("Testing center point detection...")
    
    # Create test data with clear center
    test_data = pd.DataFrame({
        'lat': [48.8566, 48.8606, 48.8516, 48.8566],  # Last point is duplicate of first
        'lon': [2.3522, 2.3376, 2.3478, 2.3522],
    })
    
    center_idx = find_center_city_point(test_data)
    
    assert center_idx in test_data.index, "Center index should be valid"
    
    print("✅ Center point detection test passed")

def create_test_excel_file():
    """Create a test Excel file for manual testing"""
    print("Creating test Excel file...")
    
    test_data = pd.DataFrame({
        'Adresse_Client': [
            '123 rue de Rivoli',
            '456 avenue des Champs-Élysées',
            '789 boulevard Saint-Germain',
            '321 rue de la Roquette',
            '654 avenue de la République'
        ],
        'Code_Postal': ['75001', '75008', '75006', '75011', '75011'],
        'Ville': ['Paris', 'Paris', 'Paris', 'Paris', 'Paris'],
        'Nom_Client': ['Client A', 'Client B', 'Client C', 'Client D', 'Client E']
    })
    
    filename = 'test_addresses.xlsx'
    test_data.to_excel(filename, index=False)
    print(f"✅ Test Excel file created: {filename}")

def run_all_tests():
    """Run all tests"""
    print("Running all tests...")
    
    try:
        test_column_detection()
        test_data_validation()
        test_route_quality_metrics()
        test_center_point_detection()
        create_test_excel_file()
        
        print("\n✅ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)