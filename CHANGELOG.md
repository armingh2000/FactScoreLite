# Changelog

All notable changes to this project will be documented in this file.

<!-- ## [Unreleased] -->

## v 0.1.0 - 2024-03-30

### Added

- Atomic fact generator class.
- OpenAI class for API call.

## v 0.2.0 - 2024-04-02

### Added

- Tests for atomic_facts and openai_agent.
- Github workflows.

## v 0.2.1 - 2024-04-02

### Changed

- Moved OpenAI configs to configs.py.

## v 0.3.0 - 2024-04-04

### Added

- Add Scorer class to scorer.py (scores atomic facts based on a knowledge source).
- Add tests for Scorer.
- Add docstrings for Scorer and atomic_facts.

## v 0.3.1 - 2024-04-04

### Changed

- Change scorer.py name to fact_scorer.py

## v 0.4.0 - 2024-04-05

### Added

- Add FactScore class for evaluating generations based on a list of knowledge source.

### Changed

- Change FactScorer class output to include GPT output for dumping.

## v 1.0.0 - 2024-04-11

### Added

- Add StateHandler for dumping in FactScore.
- Add FactScore tests.
- Add docstrings.
- Add documentation.

### Changed

- Renamed test_scorer to test_fact_scorer.

<!--
### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security
-->
