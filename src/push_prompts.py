"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)

SIMPLIFICADO: Código mais limpo e direto ao ponto.
"""

import os
import sys
from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, check_env_vars, print_section_header

load_dotenv()


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Faz push do prompt otimizado para o LangSmith Hub (PÚBLICO).

    Args:
        prompt_name: Nome do prompt
        prompt_data: Dados do prompt

    Returns:
        True se sucesso, False caso contrário
    """
    try:
        # Extrair dados do prompt
        description = prompt_data.get("description", "")
        system_prompt = prompt_data.get("system_prompt", "")
        user_prompt = prompt_data.get("user_prompt", "{bug_report}")
        tags = prompt_data.get("tags", [])
        techniques = prompt_data.get("techniques_applied", [])
        
        # Criar o template
        full_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt)
        ])
        
        # Nome completo no hub: {user}/{prompt_name}
        # O username deve vir do .env
        hub_username = os.getenv("USERNAME_LANGSMITH_HUB")
        if not hub_username:
            print("❌ ERRO: USERNAME_LANGSMITH_HUB não configurado no .env")
            return False
            
        full_hub_path = f"{hub_username}/{prompt_name}"
        
        print(f"Enviando prompt para o hub: {full_hub_path}...")
        
        # Faz o push
        # Nota: Por padrão, o hub.push torna o prompt público se não especificado o contrário
        # mas a API do hub.push do langchain costuma ser simples
        hub.push(full_hub_path, full_prompt, new_repo_is_public=True)
        
        print(f"✓ Prompt publicado com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao fazer push do prompt: {e}")
        return False


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt.

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros
    """
    from utils import validate_prompt_structure
    return validate_prompt_structure(prompt_data)


def main():
    """Função principal"""
    print_section_header("FASE 3: PUSH DE PROMPTS PARA O LANGSMITH HUB")
    
    # Verificar variáveis de ambiente
    if not check_env_vars(["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB"]):
        return 1
        
    # Carregar prompt v2
    prompt_path = "prompts/bug_to_user_story_v2.yml"
    payload = load_yaml(prompt_path)
    
    if not payload or "bug_to_user_story_v2" not in payload:
        print(f"❌ Erro: Prompt não encontrado em {prompt_path}")
        return 1
        
    prompt_data = payload["bug_to_user_story_v2"]
    
    # Validar
    is_valid, errors = validate_prompt(prompt_data)
    if not is_valid:
        print("❌ O prompt possui erros de validação:")
        for error in errors:
            print(f"   - {error}")
        return 1
        
    # Push
    success = push_prompt_to_langsmith("bug_to_user_story_v2", prompt_data)
    
    if success:
        print("\n✅ Etapa de Push concluída com sucesso!")
        return 0
    else:
        print("\n❌ Falha na etapa de Push.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
