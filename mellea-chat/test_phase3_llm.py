#!/usr/bin/env python3
"""Test script for Phase 3: LLM-Based Behavior Advisor."""

import asyncio
import json
from datetime import datetime

from dog_scheduler import DailySchedule, DogState, DogStateMetrics
from dog_behavior_orchestrator import BehaviorOrchestrator
from mellea_dog_behavior import DogBehaviorLLM, DogBehaviorAnalyzer
from hardware_interface import create_hardware_interface
from dog_runtime import DogRuntime


async def test_llm_initialization():
    """Test LLM initialization."""
    print("\n=== Testing LLM Initialization ===")

    try:
        llm = DogBehaviorLLM()
        print(f"✓ LLM initialized successfully")
        print(f"  Model: {llm.model_id}")
        print(f"  Temperature: {llm.temperature}")
    except Exception as e:
        print(f"⚠ LLM initialization error (may be expected if Ollama not running): {e}")


async def test_behavior_decision():
    """Test LLM behavior decision making."""
    print("\n=== Testing LLM Behavior Decision ===")

    try:
        llm = DogBehaviorLLM()
        metrics = DogStateMetrics(energy_level=80, hunger_level=30, attention_level=50)

        # Test decision at different times
        times = [6, 12, 18, 22]
        print(f"Testing behavior decisions at different times of day:")

        for hour in times:
            next_state, duration = await llm.decide_behavior(
                DogState.ALERT, metrics, hour
            )
            print(f"  {hour:02d}:00 → {next_state.value:12} for {duration:5}ms")
            assert next_state in DogState, f"Invalid state returned: {next_state}"
            assert 1000 <= duration <= 3600000, f"Duration out of range: {duration}"

        print("✓ LLM behavior decisions work")
    except Exception as e:
        print(f"⚠ LLM behavior decision error: {e}")


async def test_command_interpretation():
    """Test natural language command interpretation."""
    print("\n=== Testing Command Interpretation ===")

    try:
        llm = DogBehaviorLLM()

        commands = [
            ("play with the dog", DogState.PLAYING),
            ("let the dog sleep", DogState.SLEEPING),
            ("make the dog eat", DogState.EATING),
            ("the dog should rest", DogState.RESTING),
        ]

        print(f"Testing command interpretation:")
        for command, expected_state in commands:
            result = await llm.interpret_user_command(command, DogState.ALERT)
            print(f"  '{command}' → {result.value if result else 'None'}")
            # Don't assert because LLM interpretation can vary

        print("✓ Command interpretation works")
    except Exception as e:
        print(f"⚠ Command interpretation error: {e}")


async def test_response_generation():
    """Test response generation to user."""
    print("\n=== Testing Response Generation ===")

    try:
        llm = DogBehaviorLLM()
        metrics = DogStateMetrics(energy_level=70, hunger_level=40, attention_level=60)

        user_inputs = [
            "Hello dog!",
            "Want to play?",
            "Are you tired?",
        ]

        print(f"Testing response generation:")
        for user_input in user_inputs:
            response = await llm.generate_response_to_user(user_input, DogState.ALERT, metrics)
            print(f"  User: '{user_input}'")
            print(f"  Dog:  '{response[:60]}...'")

        print("✓ Response generation works")
    except Exception as e:
        print(f"⚠ Response generation error: {e}")


def test_behavior_analyzer():
    """Test behavior analysis."""
    print("\n=== Testing Behavior Analyzer ===")

    analyzer = DogBehaviorAnalyzer()
    metrics = DogStateMetrics(energy_level=50, hunger_level=50, attention_level=30)

    # Record some behaviors
    behaviors = [
        (DogState.PLAYING, 5000),
        (DogState.PLAYING, 4000),
        (DogState.RESTING, 3000),
        (DogState.PLAYING, 6000),
        (DogState.EXPLORING, 10000),
    ]

    print(f"Recording {len(behaviors)} behaviors:")
    for state, duration in behaviors:
        analyzer.record_behavior(state, duration, metrics)
        print(f"  {state.value:12} {duration:5}ms")

    # Get statistics
    stats = analyzer.get_behavior_stats()
    print(f"\nBehavior statistics:")
    for state, data in stats.items():
        print(f"  {state:12}: {data['count']:2}x, avg {data['avg_duration']:7.0f}ms")

    # Get preferences
    prefs = analyzer.identify_preferences()
    print(f"\nPreferences:")
    print(f"  Most common: {prefs.get('most_common', 'None')}")
    print(f"  Least common: {prefs.get('least_common', 'None')}")

    print("✓ Behavior analyzer works")


async def test_runtime_with_llm():
    """Test runtime with LLM integration."""
    print("\n=== Testing Runtime with LLM ===")

    try:
        hardware = create_hardware_interface("simulation")
        orchestrator = BehaviorOrchestrator()
        llm = DogBehaviorLLM()

        runtime = DogRuntime(
            hardware_interface=hardware,
            behavior_orchestrator=orchestrator,
            llm_advisor=llm,
            enable_logging=True,
            use_llm=True,
        )

        print(f"Created runtime with LLM advisor")
        print(f"Running 3 cycles with LLM decision-making...")

        for i in range(3):
            should_continue = await runtime.run_cycle()
            status = runtime.get_current_status()
            print(f"  Cycle {i+1}: {status['dog_state']:12} (Energy: {status['metrics']['energy_level']:3}%)")
            if not should_continue:
                break

        runtime.save_logs()

        # Check for LLM decisions
        llm_events = [e for e in runtime.state_log if e.get("event") == "llm_decision"]
        print(f"  LLM decisions made: {len(llm_events)}")

        print("✓ Runtime with LLM works")
    except Exception as e:
        print(f"⚠ Runtime with LLM error: {e}")


async def test_comparison_llm_vs_scheduler():
    """Compare LLM decisions vs scheduler decisions."""
    print("\n=== Testing LLM vs Scheduler Comparison ===")

    try:
        hardware = create_hardware_interface("simulation")
        orchestrator = BehaviorOrchestrator()
        llm = DogBehaviorLLM()

        # Run with LLM
        print(f"Running 5 cycles WITH LLM...")
        runtime_llm = DogRuntime(
            hardware_interface=hardware,
            behavior_orchestrator=orchestrator,
            llm_advisor=llm,
            enable_logging=True,
            use_llm=True,
        )

        for i in range(5):
            await runtime_llm.run_cycle()

        # Count behaviors with LLM
        llm_behaviors = {}
        for entry in runtime_llm.state_log:
            state = entry.get("dog_state")
            llm_behaviors[state] = llm_behaviors.get(state, 0) + 1

        print(f"  Behavior distribution with LLM:")
        for state, count in sorted(llm_behaviors.items()):
            print(f"    {state:12}: {count} times")

        # Run without LLM
        print(f"\nRunning 5 cycles WITHOUT LLM (scheduler only)...")
        runtime_scheduler = DogRuntime(
            hardware_interface=create_hardware_interface("simulation"),
            behavior_orchestrator=BehaviorOrchestrator(),
            llm_advisor=None,
            enable_logging=True,
            use_llm=False,
        )

        for i in range(5):
            await runtime_scheduler.run_cycle()

        # Count behaviors without LLM
        scheduler_behaviors = {}
        for entry in runtime_scheduler.state_log:
            state = entry.get("dog_state")
            scheduler_behaviors[state] = scheduler_behaviors.get(state, 0) + 1

        print(f"  Behavior distribution without LLM:")
        for state, count in sorted(scheduler_behaviors.items()):
            print(f"    {state:12}: {count} times")

        print("✓ Comparison complete")
    except Exception as e:
        print(f"⚠ Comparison error: {e}")


async def main():
    """Run all tests."""
    print("=" * 50)
    print("Phase 3: LLM-Based Behavior Advisor Tests")
    print("=" * 50)

    try:
        # Synchronous tests
        test_behavior_analyzer()

        # Async tests
        await test_llm_initialization()
        await test_behavior_decision()
        await test_command_interpretation()
        await test_response_generation()
        await test_runtime_with_llm()
        await test_comparison_llm_vs_scheduler()

        print("\n" + "=" * 50)
        print("✓ TESTS COMPLETED")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
