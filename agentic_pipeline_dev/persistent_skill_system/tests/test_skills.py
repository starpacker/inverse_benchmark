import unittest
import os
import shutil
import numpy as np
import sqlite3
import json
from unittest.mock import MagicMock, patch

from persistent_skill_system.storage import SkillStorage
from persistent_skill_system.manager import SkillManager
from persistent_skill_system.teacher import SkillTeacher


class TestSkillStorage(unittest.TestCase):
    """Tests for legacy skills table operations."""

    def setUp(self):
        self.db_path = "test_skills.db"
        self.storage = SkillStorage(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_add_and_get_skill(self):
        skill = {
            "name": "Test Skill",
            "principle": "Do X then Y",
            "when_to_apply": "When Z happens",
            "source": ["exp_1"],
            "embedding": [0.1] * 384
        }
        self.storage.add_skill(skill)

        retrieved = self.storage.get_relevant_skills([0.1] * 384, top_k=1)
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0]['name'], "Test Skill")
        self.assertEqual(len(retrieved[0]['embedding']), 384)

    def test_merge_skill(self):
        skill1 = {
            "name": "Test Skill 1",
            "principle": "Original Principle",
            "when_to_apply": "Condition A",
            "source": ["exp_1"],
            "embedding": [0.1] * 384
        }
        self.storage.add_skill(skill1)

        skill2 = {
            "name": "Test Skill 2",
            "principle": "Similar Principle",
            "when_to_apply": "Condition A",
            "source": ["exp_2"],
            "embedding": [0.1001] * 384
        }

        merged = self.storage.merge_skill(skill2, similarity_threshold=0.9)
        self.assertTrue(merged)

        skills = self.storage.get_all_skills()
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0]['name'], "Test Skill 1")
        self.assertIn("exp_1", skills[0]['source'])
        self.assertIn("exp_2", skills[0]['source'])


class TestKnowledgeStorage(unittest.TestCase):
    """Tests for the new layered knowledge_items table."""

    def setUp(self):
        self.db_path = "test_knowledge.db"
        self.storage = SkillStorage(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_add_and_get_knowledge_item(self):
        """Test basic CRUD for knowledge items."""
        item = {
            "id": "test-item-001",
            "name": "Test Experience",
            "type": "experience",
            "content": {
                "condition": "When data has NaN values",
                "action": "Apply imputation before processing",
                "rationale": "Prevents downstream computation errors"
            },
            "embedding": [0.5] * 384,
            "tags": ["data_preprocessing", "robustness"],
            "source_trajectories": ["exp_001"],
            "agent_scope": "General",
            "artifact_type": "experience_pattern"
        }
        result = self.storage.add_knowledge_item(item)
        self.assertTrue(result)

        # Retrieve by ID
        retrieved = self.storage.get_knowledge_by_ids(["test-item-001"])
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0]['name'], "Test Experience")
        self.assertEqual(retrieved[0]['type'], "experience")
        self.assertEqual(retrieved[0]['content']['condition'], "When data has NaN values")
        self.assertIn("data_preprocessing", retrieved[0]['tags'])

    def test_get_knowledge_items_by_type(self):
        """Test filtering knowledge items by type."""
        # Add an instance
        self.storage.add_knowledge_item({
            "name": "Plan Instance",
            "type": "instance",
            "content": "Step 1: Load data. Step 2: Process.",
            "embedding": [0.1] * 384,
            "agent_scope": "Planner",
            "artifact_type": "plan"
        })
        # Add an experience
        self.storage.add_knowledge_item({
            "name": "Debug Pattern",
            "type": "experience",
            "content": {"condition": "X", "action": "Y", "rationale": "Z"},
            "embedding": [0.2] * 384,
            "agent_scope": "General",
            "artifact_type": "experience_pattern"
        })
        # Add core knowledge
        self.storage.add_knowledge_item({
            "name": "Always Validate Input",
            "type": "core",
            "content": {"principle": "Validate all inputs", "checklist": ["Check dims", "Check dtype"]},
            "embedding": [0.3] * 384,
            "agent_scope": "General",
            "artifact_type": "principle"
        })

        instances = self.storage.get_knowledge_items(k_type='instance')
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]['name'], "Plan Instance")

        experiences = self.storage.get_knowledge_items(k_type='experience')
        self.assertEqual(len(experiences), 1)

        cores = self.storage.get_knowledge_items(k_type='core')
        self.assertEqual(len(cores), 1)

    def test_get_knowledge_items_by_agent_scope(self):
        """Test filtering by agent_scope."""
        self.storage.add_knowledge_item({
            "name": "Coder Code", "type": "instance",
            "content": "import numpy", "embedding": [0.1] * 384,
            "agent_scope": "Coder", "artifact_type": "code"
        })
        self.storage.add_knowledge_item({
            "name": "Planner Plan", "type": "instance",
            "content": "Plan: do X", "embedding": [0.2] * 384,
            "agent_scope": "Planner", "artifact_type": "plan"
        })

        coder_items = self.storage.get_knowledge_items(k_type='instance', agent_scope='Coder')
        self.assertEqual(len(coder_items), 1)
        self.assertEqual(coder_items[0]['name'], "Coder Code")

    def test_search_knowledge_vector(self):
        """Test vector search with cosine similarity."""
        # Add items with known embeddings
        self.storage.add_knowledge_item({
            "name": "Similar Item",
            "type": "experience",
            "content": {"condition": "A", "action": "B", "rationale": "C"},
            "embedding": [0.9] * 384,  # Close to query
            "agent_scope": "General",
            "artifact_type": "experience_pattern"
        })
        self.storage.add_knowledge_item({
            "name": "Dissimilar Item",
            "type": "experience",
            "content": {"condition": "X", "action": "Y", "rationale": "Z"},
            "embedding": [-0.9] * 384,  # Opposite to query
            "agent_scope": "General",
            "artifact_type": "experience_pattern"
        })

        results = self.storage.search_knowledge([0.8] * 384, k_type='experience', top_k=2)
        self.assertEqual(len(results), 2)
        # Similar item should be ranked first
        self.assertEqual(results[0]['name'], "Similar Item")

    def test_credit_score_update_success(self):
        """Test that success increments credit score."""
        item_id = "credit-test-001"
        self.storage.add_knowledge_item({
            "id": item_id,
            "name": "Credit Test",
            "type": "experience",
            "content": {"condition": "A", "action": "B", "rationale": "C"},
            "embedding": [0.1] * 384,
            "credit_score": 1.0
        })

        self.storage.update_knowledge_usage(item_id, success=True)

        items = self.storage.get_knowledge_by_ids([item_id])
        self.assertEqual(len(items), 1)
        self.assertAlmostEqual(items[0]['credit_score'], 1.1, places=5)
        self.assertEqual(items[0]['usage_count'], 1)

    def test_credit_score_update_failure(self):
        """Test that failure decrements credit score."""
        item_id = "credit-test-002"
        self.storage.add_knowledge_item({
            "id": item_id,
            "name": "Credit Fail Test",
            "type": "experience",
            "content": {"condition": "A", "action": "B", "rationale": "C"},
            "embedding": [0.1] * 384,
            "credit_score": 1.0
        })

        self.storage.update_knowledge_usage(item_id, success=False)

        items = self.storage.get_knowledge_by_ids([item_id])
        self.assertAlmostEqual(items[0]['credit_score'], 0.8, places=5)

    def test_auto_archive_on_low_credit(self):
        """Test that items with credit < -0.5 get archived."""
        item_id = "archive-test-001"
        self.storage.add_knowledge_item({
            "id": item_id,
            "name": "Bad Knowledge",
            "type": "experience",
            "content": {"condition": "A", "action": "B", "rationale": "C"},
            "embedding": [0.1] * 384,
            "credit_score": -0.4  # Close to threshold
        })

        # This should push it to -0.6, triggering archive
        self.storage.update_knowledge_usage(item_id, success=False)

        items = self.storage.get_knowledge_by_ids([item_id])
        self.assertEqual(items[0]['status'], 'archived')

    def test_json_fields_parsed_correctly(self):
        """Test that JSON text fields are properly deserialized to lists."""
        self.storage.add_knowledge_item({
            "id": "json-test-001",
            "name": "JSON Field Test",
            "type": "core",
            "content": {"principle": "Test"},
            "embedding": [0.1] * 384,
            "tags": ["tag1", "tag2"],
            "preconditions": ["pre1"],
            "source_experience_ids": ["exp-1", "exp-2"],
            "contributed_to_ck_ids": []
        })

        items = self.storage.get_knowledge_items(k_type='core')
        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0]['tags'], list)
        self.assertEqual(items[0]['tags'], ["tag1", "tag2"])
        self.assertIsInstance(items[0]['preconditions'], list)
        self.assertEqual(items[0]['preconditions'], ["pre1"])
        self.assertIsInstance(items[0]['source_experience_ids'], list)
        self.assertEqual(len(items[0]['source_experience_ids']), 2)
        self.assertIsInstance(items[0]['contributed_to_ck_ids'], list)
        self.assertEqual(items[0]['contributed_to_ck_ids'], [])


class TestSkillTeacher(unittest.TestCase):
    """Tests for the Teacher's layered trajectory analysis."""

    def test_instance_extraction_success_trajectory(self):
        """Test rule-based instance extraction from a successful trajectory."""
        mock_client = MagicMock()
        # Mock the LLM response for experience extraction
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "experiences": [
                {
                    "name": "Handling Shape Mismatch",
                    "content": {
                        "condition": "When input arrays have incompatible shapes",
                        "action": "Reshape or pad to compatible dimensions",
                        "rationale": "Prevents broadcasting errors"
                    },
                    "tags": ["debugging", "numpy"]
                }
            ]
        })
        mock_client.chat.completions.create.return_value = mock_response

        teacher = SkillTeacher(mock_client, "test-model")

        trajectory = {
            "task_name": "Solve Inverse Problem",
            "task_desc": "Reconstruct image from measurements",
            "outcome": "success",
            "final_reward": 0.95,
            "final_plan": "Step 1: Load data. Step 2: Build forward model. Step 3: Run ADMM.",
            "final_skeleton": "class Solver:\n    def __init__(self): pass\n    def solve(self): pass",
            "final_code": "import numpy as np\ndef solve(data):\n    return np.linalg.solve(A, data)",
            "steps": [
                {
                    "role": "Planner", "iteration": 1,
                    "output": {"plan": "Load data and process"}
                },
                {
                    "role": "Architect", "iteration": 1,
                    "output": {"skeleton": "class Solver: pass"}
                },
                {
                    "role": "Judge", "iteration": 1,
                    "output": {
                        "full_judgement_analysis": "Code runs correctly. Metrics are good.",
                        "ticket": None
                    }
                }
            ]
        }

        results = teacher.analyze_trajectory_layered(trajectory)

        # Check instances were extracted
        self.assertIn('instances', results)
        self.assertIn('experiences', results)

        # Should have: Planner plan, Architect skeleton, Coder code, Judge feedback = 4 instances
        instance_scopes = [inst['agent_scope'] for inst in results['instances']]
        self.assertIn('Planner', instance_scopes)
        self.assertIn('Architect', instance_scopes)
        self.assertIn('Coder', instance_scopes)
        self.assertIn('Judge', instance_scopes)

        # Check Planner instance content
        planner_inst = [i for i in results['instances'] if i['agent_scope'] == 'Planner'][0]
        self.assertEqual(planner_inst['artifact_type'], 'plan')
        self.assertIn("Load data", planner_inst['content'])

        # Check experiences from LLM
        self.assertEqual(len(results['experiences']), 1)
        self.assertEqual(results['experiences'][0]['name'], "Handling Shape Mismatch")

    def test_instance_extraction_failure_trajectory(self):
        """Test that instances are NOT extracted from failed trajectories."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "experiences": [
                {
                    "name": "Debugging Timeout",
                    "content": {
                        "condition": "When execution exceeds time limit",
                        "action": "Optimize inner loop or reduce iterations",
                        "rationale": "Prevents hanging"
                    },
                    "tags": ["performance"]
                }
            ]
        })
        mock_client.chat.completions.create.return_value = mock_response

        teacher = SkillTeacher(mock_client, "test-model")

        trajectory = {
            "task_name": "Failed Task",
            "task_desc": "This task failed",
            "outcome": "failure",
            "final_plan": "Some plan",
            "final_code": "broken code",
            "steps": []
        }

        results = teacher.analyze_trajectory_layered(trajectory)

        # No instances from failure
        self.assertEqual(len(results['instances']), 0)
        # But experiences CAN be extracted from failures
        self.assertEqual(len(results['experiences']), 1)

    def test_architect_fallback_to_steps(self):
        """Test that Architect instance falls back to step output when no top-level skeleton."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({"experiences": []})
        mock_client.chat.completions.create.return_value = mock_response

        teacher = SkillTeacher(mock_client, "test-model")

        trajectory = {
            "task_name": "Fallback Test",
            "task_desc": "Test fallback",
            "outcome": "success",
            # No final_skeleton at top level
            "steps": [
                {"role": "Architect", "iteration": 1, "output": {"skeleton": "class A: pass"}},
                {"role": "Architect", "iteration": 2, "output": {"skeleton": "class B(A): pass"}},
            ]
        }

        results = teacher.analyze_trajectory_layered(trajectory)

        arch_instances = [i for i in results['instances'] if i['agent_scope'] == 'Architect']
        self.assertEqual(len(arch_instances), 1)
        # Should pick the LAST architect step (iteration 2)
        self.assertIn("class B", arch_instances[0]['content'])

    def test_summarize_steps(self):
        """Test step summarization for prompt construction."""
        teacher = SkillTeacher(MagicMock(), "test-model")

        steps = [
            {"role": "Planner", "iteration": 1, "output": {"plan": "Do X then Y"}},
            {"role": "Execution", "iteration": 1, "output": {"success": True, "metrics": {"psnr": 25.0}}},
            {"role": "Execution", "iteration": 2, "output": {"success": False, "stderr": "ImportError: No module named 'foo'"}},
        ]

        summary = teacher._summarize_steps(steps)
        self.assertIn("[Iter 1] Planner:", summary)
        self.assertIn("Do X then Y", summary)
        self.assertIn("Success. Metrics:", summary)
        self.assertIn("Failed. Error:", summary)
        self.assertIn("ImportError", summary)

    def test_summarize_empty_steps(self):
        teacher = SkillTeacher(MagicMock(), "test-model")
        self.assertEqual(teacher._summarize_steps([]), "No steps recorded.")


class TestSkillManager(unittest.TestCase):
    """Tests for the SkillManager orchestration."""

    def setUp(self):
        self.db_path = "test_manager.db"
        self.mock_client = MagicMock()
        self.manager = SkillManager(self.db_path, self.mock_client, "test-model")

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_deinstantiate_url(self):
        text = "Check http://example.com/api/v1?id=12345678"
        cleaned = self.manager._deinstantiate(text)
        self.assertIn("{url}", cleaned)
        self.assertNotIn("http://", cleaned)

    def test_deinstantiate_uuid(self):
        text = "The ID is 12345678-1234-1234-1234-123456789abc"
        cleaned = self.manager._deinstantiate(text)
        self.assertIn("{uuid}", cleaned)
        self.assertNotIn("12345678-", cleaned)

    def test_deinstantiate_path(self):
        text = "Save to /home/user/data.txt"
        cleaned = self.manager._deinstantiate(text)
        self.assertIn("{path}", cleaned)
        self.assertNotIn("/home/user", cleaned)

    def test_deinstantiate_number(self):
        text = "Value is 9999"
        cleaned = self.manager._deinstantiate(text)
        self.assertIn("{number}", cleaned)
        self.assertNotIn("9999", cleaned)

    def test_deinstantiate_empty(self):
        self.assertEqual(self.manager._deinstantiate(""), "")
        self.assertEqual(self.manager._deinstantiate(None), "")

    def test_distill_and_store_layered(self):
        """Test the full layered distillation pipeline."""
        # Mock LLM response for experience extraction
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "experiences": [
                {
                    "name": "Handling NaN in Optimization",
                    "content": {
                        "condition": "When ADMM produces NaN",
                        "action": "Normalize input data",
                        "rationale": "Prevents gradient explosion"
                    },
                    "tags": ["optimization", "numerical_stability"]
                }
            ]
        })
        self.mock_client.chat.completions.create.return_value = mock_response

        trajectory = {
            "task_name": "ADMM Reconstruction",
            "task_desc": "Reconstruct image using ADMM",
            "outcome": "success",
            "exp_id": "exp_test_001",
            "final_plan": "Step 1: Load. Step 2: Solve.",
            "final_code": "import numpy as np\nx = np.zeros(10)",
            "steps": [
                {"role": "Planner", "iteration": 1, "output": {"plan": "Load and solve"}},
            ]
        }

        stats = self.manager.distill_and_store(trajectory)

        # Should have stored instances (plan + code = 2) and experiences (1)
        self.assertGreaterEqual(stats['instances'], 2)  # Planner + Coder at least
        self.assertEqual(stats['experiences'], 1)

        # Verify items in DB
        all_instances = self.manager.storage.get_knowledge_items(k_type='instance')
        self.assertGreaterEqual(len(all_instances), 2)

        all_experiences = self.manager.storage.get_knowledge_items(k_type='experience')
        self.assertEqual(len(all_experiences), 1)
        self.assertEqual(all_experiences[0]['content']['condition'], "When ADMM produces NaN")

    def test_retrieve_knowledge_layered(self):
        """Test layered knowledge retrieval."""
        # Populate DB with items of different types
        self.manager.storage.add_knowledge_item({
            "name": "Core Rule",
            "type": "core",
            "content": {"principle": "Always validate", "checklist": ["Check dims"]},
            "embedding": self.manager.get_embedding("validate input data"),
            "agent_scope": "General",
            "artifact_type": "principle"
        })
        self.manager.storage.add_knowledge_item({
            "name": "Debug Pattern",
            "type": "experience",
            "content": {"condition": "Shape error", "action": "Reshape", "rationale": "Fix dims"},
            "embedding": self.manager.get_embedding("shape mismatch debugging"),
            "agent_scope": "General",
            "artifact_type": "experience_pattern"
        })
        self.manager.storage.add_knowledge_item({
            "name": "Coder Example",
            "type": "instance",
            "content": "import numpy as np",
            "embedding": self.manager.get_embedding("numpy code example"),
            "agent_scope": "Coder",
            "artifact_type": "code"
        })

        result = self.manager.retrieve_knowledge("Fix shape mismatch in numpy array", agent_role='Coder')

        self.assertIn('core', result)
        self.assertIn('experience', result)
        self.assertIn('instance', result)

    def test_format_knowledge_for_prompt(self):
        """Test that knowledge is formatted into readable prompt sections."""
        knowledge = {
            "core": [
                {
                    "name": "Validate Input",
                    "content": {"principle": "Always check array dimensions", "checklist": ["Check shape", "Check dtype"]}
                }
            ],
            "experience": [
                {
                    "name": "Shape Fix",
                    "content": {"condition": "Mismatch", "action": "Reshape", "rationale": "Compatibility"}
                }
            ],
            "instance": [
                {
                    "name": "Code Example",
                    "content": "x = np.zeros(10)",
                    "artifact_type": "code"
                }
            ]
        }

        formatted = self.manager.format_knowledge_for_prompt(knowledge)

        self.assertIn("CORE KNOWLEDGE", formatted)
        self.assertIn("Validate Input", formatted)
        self.assertIn("EXPERIENCE PATTERNS", formatted)
        self.assertIn("Shape Fix", formatted)
        self.assertIn("REFERENCE EXAMPLES", formatted)
        self.assertIn("x = np.zeros(10)", formatted)

    def test_format_empty_knowledge(self):
        """Test that empty knowledge returns empty string."""
        knowledge = {"core": [], "experience": [], "instance": []}
        self.assertEqual(self.manager.format_knowledge_for_prompt(knowledge), "")

    def test_update_scores_batch(self):
        """Test batch credit score updates."""
        ids = []
        for i in range(3):
            item_id = f"batch-{i}"
            self.manager.storage.add_knowledge_item({
                "id": item_id,
                "name": f"Item {i}",
                "type": "experience",
                "content": {"condition": "A", "action": "B", "rationale": "C"},
                "embedding": [0.1 * (i + 1)] * 384,
                "credit_score": 1.0
            })
            ids.append(item_id)

        self.manager.update_scores(ids, success=True)

        for item_id in ids:
            items = self.manager.storage.get_knowledge_by_ids([item_id])
            self.assertAlmostEqual(items[0]['credit_score'], 1.1, places=5)

    def test_get_embedding_deterministic(self):
        """Test that fallback embedding is deterministic for same input."""
        emb1 = self.manager.get_embedding("test input string")
        emb2 = self.manager.get_embedding("test input string")
        self.assertEqual(emb1, emb2)

    def test_get_embedding_dimension(self):
        """Test that embedding has correct dimension."""
        emb = self.manager.get_embedding("test")
        self.assertEqual(len(emb), self.manager.target_dim)


if __name__ == '__main__':
    unittest.main()
