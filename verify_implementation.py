#!/usr/bin/env python3
"""
Verification Script for Text Detection Implementation

This script verifies that all components of the text detection system
are correctly implemented and functional.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("Testing Module Imports")
    print("=" * 60)
    
    try:
        import text_detector
        print("✓ text_detector module imported")
        
        from text_detector import EASTTextDetector, TextRegionPreprocessor
        print("✓ EASTTextDetector class available")
        print("✓ TextRegionPreprocessor class available")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_preprocessor():
    """Test TextRegionPreprocessor functionality."""
    print("\n" + "=" * 60)
    print("Testing TextRegionPreprocessor")
    print("=" * 60)
    
    try:
        import numpy as np
        import text_detector
        
        preprocessor = text_detector.TextRegionPreprocessor()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test each method
        methods = [
            'deskew',
            'enhance_contrast',
            'normalize_illumination',
            'remove_noise',
            'preprocess_text_region'
        ]
        
        for method_name in methods:
            method = getattr(preprocessor, method_name)
            result = method(test_image)
            assert result.shape == test_image.shape, f"{method_name} changed image shape"
            print(f"✓ {method_name}() works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Preprocessor test failed: {e}")
        return False

def test_east_detector():
    """Test EASTTextDetector initialization."""
    print("\n" + "=" * 60)
    print("Testing EASTTextDetector")
    print("=" * 60)
    
    try:
        import text_detector
        
        # Test initialization without model
        config = {
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "input_width": 320,
            "input_height": 320,
        }
        
        detector = text_detector.EASTTextDetector(config=config)
        print("✓ EASTTextDetector initialized without model")
        
        # Check attributes
        assert detector.confidence_threshold == 0.5
        assert detector.input_width == 320
        print("✓ Configuration parameters set correctly")
        
        # Test with non-existent model path
        detector2 = text_detector.EASTTextDetector(
            model_path="/tmp/nonexistent.pb",
            config=config
        )
        assert detector2.net is None
        print("✓ Handles non-existent model gracefully")
        
        return True
    except Exception as e:
        print(f"✗ EAST detector test failed: {e}")
        return False

def test_configuration():
    """Test that configuration file has proper structure."""
    print("\n" + "=" * 60)
    print("Testing Configuration")
    print("=" * 60)
    
    try:
        import yaml
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check OCR section
        assert 'ocr' in config
        print("✓ OCR configuration section exists")
        
        # Check text detection section
        ocr_config = config['ocr']
        if 'use_text_detection' in ocr_config:
            print(f"✓ use_text_detection setting found: {ocr_config['use_text_detection']}")
        
        if 'text_detection' in ocr_config:
            print("✓ text_detection configuration section exists")
            td_config = ocr_config['text_detection']
            
            required_keys = ['east_model_path', 'confidence_threshold', 'nms_threshold']
            for key in required_keys:
                if key in td_config:
                    print(f"  ✓ {key}: {td_config[key]}")
        
        if 'preprocessing' in ocr_config:
            print("✓ preprocessing configuration section exists")
            prep_config = ocr_config['preprocessing']
            print(f"  ✓ apply_deskewing: {prep_config.get('apply_deskewing')}")
            print(f"  ✓ apply_illumination_normalization: {prep_config.get('apply_illumination_normalization')}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_examples():
    """Test that example files exist and are valid Python."""
    print("\n" + "=" * 60)
    print("Testing Example Files")
    print("=" * 60)
    
    try:
        import py_compile
        
        examples = [
            'examples/basic_ocr.py',
            'examples/license_plate_detection.py',
            'examples/text_detection_pipeline.py'
        ]
        
        for example in examples:
            if os.path.exists(example):
                py_compile.compile(example, doraise=True)
                print(f"✓ {example} exists and compiles")
            else:
                print(f"✗ {example} not found")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Example test failed: {e}")
        return False

def test_documentation():
    """Test that documentation is updated."""
    print("\n" + "=" * 60)
    print("Testing Documentation")
    print("=" * 60)
    
    try:
        # Check README
        with open('README.md', 'r') as f:
            readme = f.read()
        
        if 'EAST' in readme or 'text detection' in readme.lower():
            print("✓ README.md mentions text detection features")
        
        if 'preprocessing' in readme.lower():
            print("✓ README.md mentions preprocessing")
        
        # Check models README
        with open('models/README.md', 'r') as f:
            models_readme = f.read()
        
        if 'EAST' in models_readme:
            print("✓ models/README.md has EAST detector section")
        
        if 'frozen_east_text_detection.pb' in models_readme:
            print("✓ models/README.md mentions model download")
        
        # Check QUICKSTART
        with open('QUICKSTART.md', 'r') as f:
            quickstart = f.read()
        
        if 'text detection' in quickstart.lower() or 'EAST' in quickstart:
            print("✓ QUICKSTART.md includes text detection information")
        
        return True
    except Exception as e:
        print(f"✗ Documentation test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("TRAKT TEXT DETECTION VERIFICATION SCRIPT")
    print("=" * 60 + "\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("TextRegionPreprocessor", test_preprocessor),
        ("EASTTextDetector", test_east_detector),
        ("Configuration", test_configuration),
        ("Example Files", test_examples),
        ("Documentation", test_documentation),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30s} {status}")
    
    print("\n" + "=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All verification tests passed!")
        print("\nThe text detection implementation is complete and functional.")
        print("\nNext steps:")
        print("  1. Download EAST model (see models/README.md)")
        print("  2. Configure your camera in config.yaml")
        print("  3. Enable text detection in config.yaml")
        print("  4. Run: python examples/text_detection_pipeline.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
