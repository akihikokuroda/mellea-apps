#!/usr/bin/env python3
"""Test script for Phase 4: User Interface."""

import asyncio
from dog_scheduler import DogState, DogStateMetrics
from dog_command_parser import CommandParser, ResponseGenerator, ConversationManager, InteractiveSession


def test_command_parser():
    """Test command parsing."""
    print("\n=== Testing Command Parser ===")

    parser = CommandParser()

    # Test action commands
    print(f"\nAction commands:")
    action_tests = [
        ("make the dog play", "action", DogState.PLAYING),
        ("let the dog sleep", "action", DogState.SLEEPING),
        ("dog should rest", "action", DogState.RESTING),
        ("feed the dog", "action", DogState.EATING),
        ("dog explore", "action", DogState.EXPLORING),
    ]

    for input_text, expected_type, expected_state in action_tests:
        parsed = parser.parse(input_text)
        assert parsed.command_type == expected_type, f"Expected {expected_type}, got {parsed.command_type}"
        assert parsed.target_state == expected_state, f"Expected {expected_state}, got {parsed.target_state}"
        print(f"  ✓ '{input_text}' → {expected_state.value}")

    # Test query commands
    print(f"\nQuery commands:")
    query_tests = [
        ("is the dog tired?", "query", "energy"),
        ("is the dog hungry?", "query", "hunger"),
        ("dog status", "status"),
        ("is the dog's energy high?", "query", "energy"),
    ]

    for input_text, expected_type, *expected_query in query_tests:
        parsed = parser.parse(input_text)
        assert parsed.command_type == expected_type, f"Expected {expected_type}, got {parsed.command_type}"
        if expected_query:
            assert parsed.query_type == expected_query[0], f"Expected {expected_query[0]}, got {parsed.query_type}"
        print(f"  ✓ '{input_text}' → {expected_type}")

    # Test special commands
    print(f"\nSpecial commands:")
    special_tests = [
        ("help", "help"),
        ("stop", "stop"),
        ("quit", "stop"),
    ]

    for input_text, expected_type in special_tests:
        parsed = parser.parse(input_text)
        assert parsed.command_type == expected_type, f"Expected {expected_type}, got {parsed.command_type}"
        print(f"  ✓ '{input_text}' → {expected_type}")

    print("✓ Command parser tests passed")


def test_response_generator():
    """Test response generation."""
    print("\n=== Testing Response Generator ===")

    responder = ResponseGenerator()

    # Test action confirmations
    print(f"\nAction confirmations:")
    for state in DogState:
        response = responder.get_action_confirmation(state)
        assert response, f"No response for {state}"
        assert len(response) > 0, f"Empty response for {state}"
        print(f"  ✓ {state.value:12} → '{response[:50]}...'")

    # Test query responses
    print(f"\nQuery responses:")
    metrics_high = DogStateMetrics(energy_level=90, hunger_level=80, attention_level=70)
    metrics_low = DogStateMetrics(energy_level=20, hunger_level=30, attention_level=10)

    print(f"  High energy: {responder.get_query_response('energy', metrics_high)}")
    print(f"  Low energy:  {responder.get_query_response('energy', metrics_low)}")
    print(f"  High hunger: {responder.get_query_response('hunger', metrics_high)}")
    print(f"  Low hunger:  {responder.get_query_response('hunger', metrics_low)}")

    # Test status response
    print(f"\nStatus response:")
    status = responder.get_status_response(DogState.PLAYING, metrics_high)
    assert "PLAYING" in status
    assert "Energy" in status
    print(f"  ✓ Status includes state and metrics")

    print("✓ Response generator tests passed")


def test_conversation_manager():
    """Test conversation management."""
    print("\n=== Testing Conversation Manager ===")

    manager = ConversationManager(max_history=10)

    # Add exchanges
    print(f"\nAdding 5 exchanges:")
    for i in range(5):
        manager.add_exchange(
            f"user input {i}",
            f"dog response {i}",
            "action" if i % 2 == 0 else "query"
        )
        print(f"  ✓ Exchange {i+1} added")

    # Check history
    history = manager.get_history()
    assert len(history) == 5, f"Expected 5 exchanges, got {len(history)}"
    print(f"✓ History length: {len(history)}")

    # Check summary
    summary = manager.get_summary()
    assert summary["total_exchanges"] == 5
    assert "action" in summary["command_types"]
    assert summary["command_types"]["action"] == 3
    print(f"✓ Summary: {summary['total_exchanges']} exchanges, {summary['command_types']}")

    print("✓ Conversation manager tests passed")


async def test_interactive_session():
    """Test interactive session."""
    print("\n=== Testing Interactive Session ===")

    session = InteractiveSession()

    # Test processing commands
    print(f"\nProcessing commands:")
    metrics = DogStateMetrics(energy_level=70, hunger_level=40, attention_level=50)

    test_commands = [
        ("make the dog play", "action"),
        ("is the dog tired?", "query"),
        ("dog status", "status"),
        ("help", "help"),
    ]

    for user_input, expected_type in test_commands:
        cmd_type, target, response = session.process_command(
            user_input,
            DogState.ALERT,
            metrics
        )
        assert cmd_type == expected_type, f"Expected {expected_type}, got {cmd_type}"
        assert len(response) > 0, f"Empty response for {user_input}"
        print(f"  ✓ '{user_input}' → {cmd_type}")

    # Check conversation summary
    summary = session.get_conversation_summary()
    assert summary["total_exchanges"] == len(test_commands)
    print(f"✓ Conversation recorded: {summary['total_exchanges']} exchanges")

    print("✓ Interactive session tests passed")


def test_help_text():
    """Test help text generation."""
    print("\n=== Testing Help Text ===")

    parser = CommandParser()
    help_text = parser.get_help_text()

    # Check for key sections
    assert "BEHAVIOR COMMANDS" in help_text
    assert "STATUS QUERIES" in help_text
    assert "play" in help_text.lower()
    assert "tired" in help_text.lower()

    print(f"✓ Help text contains all sections")
    print(f"✓ Help text length: {len(help_text)} characters")


async def test_invalid_commands():
    """Test handling of invalid commands."""
    print("\n=== Testing Invalid Command Handling ===")

    session = InteractiveSession()
    metrics = DogStateMetrics()

    invalid_inputs = [
        "xyzabc qwerty",
        "gibberish nonsense",
        "12345 abcde",
    ]

    print(f"Testing invalid inputs:")
    for user_input in invalid_inputs:
        cmd_type, target, response = session.process_command(
            user_input,
            DogState.ALERT,
            metrics
        )
        assert cmd_type == "invalid", f"Expected invalid, got {cmd_type}"
        assert len(response) > 0, f"No response for invalid input"
        print(f"  ✓ '{user_input}' → invalid (response: '{response[:40]}...')")

    print("✓ Invalid command handling tests passed")


async def main():
    """Run all tests."""
    print("=" * 50)
    print("Phase 4: User Interface Tests")
    print("=" * 50)

    try:
        # Synchronous tests
        test_command_parser()
        test_response_generator()
        test_conversation_manager()
        test_help_text()

        # Async tests
        await test_interactive_session()
        await test_invalid_commands()

        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
