"""
Test Suite for Knowledge Graph-Based Propaganda Detection

This script tests all components of the three-stream architecture.
"""

import json
import sys
from prerequisite_checker import PrerequisiteChecker
from knowledge_graph import KnowledgeGraph
from three_stream_architecture import ThreeStreamArchitecture


def test_configuration_loading():
    """Test that configuration file loads correctly."""
    print("Testing configuration loading...", end=" ")

    try:
        with open("propaganda_techniques.json", 'r') as f:
            config = json.load(f)

        assert 'propaganda_techniques' in config
        assert len(config['propaganda_techniques']) == 22
        assert 'hierarchical_structure' in config
        assert 'co_occurrence_clusters' in config
        assert 'mutual_exclusions' in config

        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_prerequisite_checker():
    """Test prerequisite checking for various techniques."""
    print("\nTesting prerequisite checker...")

    checker = PrerequisiteChecker()

    # Test cases with expected techniques
    test_cases = [
        {
            'text': 'Dr. Smith says this is true. Scientists agree.',
            'expected': ['Appeal to Authority'],
            'name': 'Authority detection'
        },
        {
            'text': 'Everyone knows this! Nobody disagrees!',
            'expected': ['Bandwagon'],
            'name': 'Bandwagon detection'
        },
        {
            'text': 'But what about when they did it?',
            'expected': ['Whataboutism'],
            'name': 'Whataboutism detection'
        },
        {
            'text': 'This corrupt liar is a complete fraud!',
            'expected': ['Loaded Language', 'Name Calling/Labeling'],
            'name': 'Loaded language detection'
        },
        {
            'text': 'Freedom! Justice! Democracy! Hope!',
            'expected': ['Glittering Generalities'],
            'name': 'Glittering generalities detection'
        },
        {
            'text': 'They are coming to destroy everything you love!',
            'expected': ['Appeal to Fear/Prejudice', 'Appeal to (Strong) Emotions'],
            'name': 'Fear appeal detection'
        }
    ]

    passed = 0
    failed = 0

    for test_case in test_cases:
        print(f"\n  Test: {test_case['name']}")
        print(f"  Text: \"{test_case['text']}\"")

        results = checker.check_all_prerequisites(test_case['text'], {})
        satisfied = [tech for tech, sat in results.items() if sat]

        # Check if expected techniques have prerequisites satisfied
        all_expected_found = all(tech in satisfied for tech in test_case['expected'])

        if all_expected_found:
            print(f"  ✓ PASSED - Found: {', '.join(test_case['expected'])}")
            passed += 1
        else:
            print(f"  ✗ FAILED - Expected: {test_case['expected']}, Got: {satisfied[:5]}")
            failed += 1

    print(f"\nPrerequisite Checker: {passed}/{len(test_cases)} tests passed")
    return failed == 0


def test_knowledge_graph():
    """Test knowledge graph reasoning."""
    print("\nTesting knowledge graph...")

    kg = KnowledgeGraph()

    # Test 1: Prerequisite filtering
    print("\n  Test 1: Prerequisite filtering")
    neural_preds = {tech: 0.5 for tech in kg.technique_names}
    prereq_scores = {tech: 0.0 for tech in kg.technique_names}
    prereq_scores['Loaded Language'] = 1.0  # Only this satisfied

    filtered = kg.apply_prerequisite_filtering(neural_preds, prereq_scores)

    if filtered['Loaded Language'] > 0.4 and filtered['Bandwagon'] < 0.1:
        print("  ✓ PASSED - Correctly filtered based on prerequisites")
        test1_passed = True
    else:
        print("  ✗ FAILED - Prerequisite filtering not working")
        test1_passed = False

    # Test 2: Co-occurrence boosting
    print("\n  Test 2: Co-occurrence boosting")
    neural_preds = {tech: 0.1 for tech in kg.technique_names}
    neural_preds['Loaded Language'] = 0.8
    neural_preds['Name Calling/Labeling'] = 0.2  # Should be boosted

    boosted = kg.apply_co_occurrence_boost(neural_preds, threshold=0.5)

    if boosted['Name Calling/Labeling'] > neural_preds['Name Calling/Labeling']:
        print(f"  ✓ PASSED - Name Calling boosted from {neural_preds['Name Calling/Labeling']:.2f} to {boosted['Name Calling/Labeling']:.2f}")
        test2_passed = True
    else:
        print("  ✗ FAILED - Co-occurrence boost not working")
        test2_passed = False

    # Test 3: Hierarchical consistency
    print("\n  Test 3: Hierarchical consistency")
    neural_preds = {tech: 0.1 for tech in kg.technique_names}
    neural_preds['Ethos'] = 0.9  # High parent
    neural_preds['Smears'] = 0.2  # Low child

    consistent = kg.enforce_hierarchical_consistency(neural_preds)

    if consistent['Smears'] > neural_preds['Smears']:
        print(f"  ✓ PASSED - Child technique boosted to match parent")
        test3_passed = True
    else:
        print("  ✗ FAILED - Hierarchical consistency not working")
        test3_passed = False

    all_passed = test1_passed and test2_passed and test3_passed
    print(f"\nKnowledge Graph: {'All tests passed' if all_passed else 'Some tests failed'}")
    return all_passed


def test_three_stream_architecture():
    """Test full three-stream architecture."""
    print("\nTesting three-stream architecture...")

    detector = ThreeStreamArchitecture(use_llm=False)

    # Test case
    meme_text = """
    These corrupt politicians are destroying our freedom!
    Everyone knows this is true. Don't be fooled!
    Our patriotic duty is to fight back!
    """

    image_features = {
        'entities': [{'text': 'politicians', 'type': 'GROUP'}],
        'symbols': ['flag'],
        'emotion': 'anger',
        'faces': 0,
        'scene': 'protest',
        'objects': ['flag', 'crowd']
    }

    # Simulate neural predictions
    neural_predictions = {tech: 0.1 for tech in detector.technique_names}
    neural_predictions['Loaded Language'] = 0.75
    neural_predictions['Appeal to (Strong) Emotions'] = 0.70
    neural_predictions['Flag-waving'] = 0.65
    neural_predictions['Name Calling/Labeling'] = 0.60
    neural_predictions['Bandwagon'] = 0.45

    print("\n  Running prediction pipeline...")

    try:
        result = detector.predict(
            text=meme_text,
            image_features=image_features,
            neural_predictions=neural_predictions,
            return_explanations=False
        )

        print(f"\n  Detected techniques: {len(result['detected_techniques'])}")
        for tech in result['detected_techniques'][:5]:
            neural = result['stream_predictions']['neural'][tech]
            kg = result['stream_predictions']['knowledge_graph'][tech]
            final = result['final_predictions'][tech]
            print(f"    - {tech}: Neural={neural:.2f}, KG={kg:.2f}, Final={final:.2f}")

        # Verify that we got reasonable results
        if len(result['detected_techniques']) > 0:
            print("\n  ✓ PASSED - Pipeline executed successfully")
            return True
        else:
            print("\n  ✗ FAILED - No techniques detected")
            return False

    except Exception as e:
        print(f"\n  ✗ FAILED - Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")

    checker = PrerequisiteChecker()
    kg = KnowledgeGraph()

    # Test 1: Empty text
    print("\n  Test 1: Empty text")
    try:
        results = checker.check_all_prerequisites("", {})
        print("  ✓ PASSED - Handled empty text")
        test1_passed = True
    except Exception as e:
        print(f"  ✗ FAILED - Exception on empty text: {e}")
        test1_passed = False

    # Test 2: Very long text
    print("\n  Test 2: Very long text")
    try:
        long_text = "word " * 1000
        results = checker.check_all_prerequisites(long_text, {})
        print("  ✓ PASSED - Handled long text")
        test2_passed = True
    except Exception as e:
        print(f"  ✗ FAILED - Exception on long text: {e}")
        test2_passed = False

    # Test 3: Special characters
    print("\n  Test 3: Special characters")
    try:
        special_text = "Test!@#$%^&*()[]{}|\\<>?/~`"
        results = checker.check_all_prerequisites(special_text, {})
        print("  ✓ PASSED - Handled special characters")
        test3_passed = True
    except Exception as e:
        print(f"  ✗ FAILED - Exception on special characters: {e}")
        test3_passed = False

    # Test 4: Missing image features
    print("\n  Test 4: Missing image features")
    try:
        results = checker.check_all_prerequisites("Test text", None)
        print("  ✓ PASSED - Handled missing image features")
        test4_passed = True
    except Exception as e:
        print(f"  ✗ FAILED - Exception on missing image features: {e}")
        test4_passed = False

    all_passed = test1_passed and test2_passed and test3_passed and test4_passed
    print(f"\nEdge Cases: {'All tests passed' if all_passed else 'Some tests failed'}")
    return all_passed


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("KNOWLEDGE GRAPH ARCHITECTURE TEST SUITE")
    print("=" * 80)

    results = []

    # Run tests
    results.append(("Configuration Loading", test_configuration_loading()))
    results.append(("Prerequisite Checker", test_prerequisite_checker()))
    results.append(("Knowledge Graph", test_knowledge_graph()))
    results.append(("Three-Stream Architecture", test_three_stream_architecture()))
    results.append(("Edge Cases", test_edge_cases()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total = len(results)
    passed = sum(1 for _, result in results if result)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<50} {status}")

    print("=" * 80)
    print(f"TOTAL: {passed}/{total} test groups passed")
    print("=" * 80)

    if passed == total:
        print("\n🎉 All tests passed! The implementation is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test group(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
