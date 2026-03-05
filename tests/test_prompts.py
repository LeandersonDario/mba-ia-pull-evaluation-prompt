"""
Testes automatizados para validação de prompts.
"""
import pytest
import yaml
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure

def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class TestPrompts:
    @pytest.fixture
    def prompts_v2(self):
        """Fixture para carregar os prompts otimizados."""
        file_path = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
        if not file_path.exists():
            pytest.fail(f"Arquivo {file_path} não encontrado. Crie-o primeiro.")
        return load_prompts(str(file_path))

    @pytest.fixture
    def prompt_v2_data(self, prompts_v2):
        """Fixture para obter os dados do prompt v2."""
        if "bug_to_user_story_v2" not in prompts_v2:
            pytest.fail("Chave 'bug_to_user_story_v2' não encontrada no YAML.")
        return prompts_v2["bug_to_user_story_v2"]

    def test_prompt_has_system_prompt(self, prompt_v2_data):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        assert "system_prompt" in prompt_v2_data
        assert len(prompt_v2_data["system_prompt"].strip()) > 0

    def test_prompt_has_role_definition(self, prompt_v2_data):
        """Verifica se o prompt define uma persona (ex: "Você é um Product Manager")."""
        system_prompt = prompt_v2_data["system_prompt"].lower()
        role_keywords = ["product manager", "pm", "especialista", "persona", "você é", "atue como", "responsável"]
        assert any(keyword in system_prompt for keyword in role_keywords)

    def test_prompt_mentions_format(self, prompt_v2_data):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        system_prompt = prompt_v2_data["system_prompt"].lower()
        format_keywords = ["markdown", "como um", "eu quero", "para que", "user story"]
        assert any(keyword in system_prompt for keyword in format_keywords)

    def test_prompt_has_few_shot_examples(self, prompt_v2_data):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        system_prompt = prompt_v2_data["system_prompt"].lower()
        example_keywords = ["exemplo", "example", "entrada:", "saída:", "input:", "output:"]
        assert any(keyword in system_prompt for keyword in example_keywords)

    def test_prompt_no_todos(self, prompt_v2_data):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        system_prompt = prompt_v2_data["system_prompt"].upper()
        # Verificar especificamente por [TODO] ou a palavra TODO com espaços around
        # para evitar falsos positivos com "todos", "método", etc.
        assert "[TODO]" not in system_prompt
        assert " TODO " not in f" {system_prompt} "

    def test_minimum_techniques(self, prompt_v2_data):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        techniques = prompt_v2_data.get("techniques_applied", [])
        assert isinstance(techniques, list)
        assert len(techniques) >= 2

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])