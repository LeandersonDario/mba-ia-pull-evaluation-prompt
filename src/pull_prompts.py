"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull dos prompts do Hub
3. Salva localmente em prompts/bug_to_user_story_v1.yml

SIMPLIFICADO: Usa serialização nativa do LangChain para extrair prompts.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain import hub
from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()


def pull_prompts_from_langsmith():
    """
    Busca o prompt base do LangSmith Hub e salva localmente.
    """
    # Nome do prompt original para o desafio
    # Note: Mantemos o nome original 'leonanluppi' para o pull da v1 conforme requisitos
    # mas poderíamos parametrizar se necessário.
    original_prompt_name = "leonanluppi/bug_to_user_story_v1"
    
    print(f"Buscando prompt '{original_prompt_name}' do hub...")
    
    try:
        # Faz o pull do prompt do hub
        prompt = hub.pull(original_prompt_name)
        
        # Extrair a estrutura para salvar em YAML
        # O objeto retornado por hub.pull é um ChatPromptTemplate
        
        # Estrutura simplificada para o desafio
        prompt_data = {
            "bug_to_user_story_v1": {
                "description": "Prompt para converter relatos de bugs em User Stories",
                "system_prompt": "",
                "user_prompt": "",
                "version": "v1",
                "created_at": "2025-01-15",
                "tags": ["bug-analysis", "user-story", "product-management"]
            }
        }
        
        # Extrair mensagens
        for message in prompt.messages:
            if hasattr(message, 'prompt') and hasattr(message.prompt, 'template'):
                template = message.prompt.template
            elif hasattr(message, 'content'):
                template = message.content
            else:
                continue
                
            if "system" in str(type(message)).lower():
                prompt_data["bug_to_user_story_v1"]["system_prompt"] = template
            elif "human" in str(type(message)).lower() or "user" in str(type(message)).lower():
                prompt_data["bug_to_user_story_v1"]["user_prompt"] = template

        # Salvar localmente
        output_path = "prompts/bug_to_user_story_v1.yml"
        if save_yaml(prompt_data, output_path):
            print(f"✓ Prompt salvo com sucesso em: {output_path}")
            return True
        else:
            print(f"❌ Erro ao salvar o prompt em {output_path}")
            return False
            
    except Exception as e:
        print(f"❌ Erro ao fazer pull do prompt: {e}")
        return False


def main():
    """Função principal"""
    print_section_header("FASE 1: PULL DE PROMPTS DO LANGSMITH")
    
    # Verificar variáveis de ambiente
    if not check_env_vars(["LANGSMITH_API_KEY"]):
        return 1
        
    success = pull_prompts_from_langsmith()
    
    if success:
        print("\n✅ Etapa de Pull concluída com sucesso!")
        return 0
    else:
        print("\n❌ Falha na etapa de Pull.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
