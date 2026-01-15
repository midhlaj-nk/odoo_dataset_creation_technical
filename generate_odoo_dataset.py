#!/usr/bin/env python3
"""
Odoo Dataset Generator

This script parses Odoo addon modules to extract class and function information.
It outputs an Excel file with separate sheets for each module.
It can use Ollama or OpenAI-compatible LLMs to generate descriptive summaries.

Usage:
    python3 generate_odoo_dataset.py /path/to/odoo /path/to/output.xlsx [options]
"""

import ast
import os
import sys
import argparse
import csv
import logging
import json
import time
import concurrent.futures
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple

import pandas as pd
import requests
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Optional: Load environment variables if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('odoo_generator.log')
    ]
)
logger = logging.getLogger(__name__)

# --- LLM Providers ---

class LLMProvider(ABC):
    @abstractmethod
    def generate_description(self, source_code: str) -> str:
        pass

class OllamaProvider(LLMProvider):
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = f"{base_url.rstrip('/')}/api/generate"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException)
    )
    def generate_description(self, source_code: str) -> str:
        prompt = f"""
        Analyze the following Odoo Python code (function/method) and provide a concise, high-level description of what it does.
        If the code has a docstring, use it as the primary source of truth but enhance it.
        Focus on the business logic or purpose.
        Return ONLY the description text, no preamble or markdown formatting.
        
        Code:
        ```python
        {source_code}
        ```
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(self.base_url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "").strip()

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException)
    )
    def generate_description(self, source_code: str) -> str:
        prompt = f"Analyze the following Odoo Python code and provide a concise business-logic description. Return ONLY the description text.\n\nCode:\n{source_code}"
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()

# --- AST Parsing ---

class OdooASTParser:
    @staticmethod
    def get_decorators(node: ast.AST) -> List[str]:
        decorators = []
        if not hasattr(node, 'decorator_list'):
            return []
            
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(f"@{decorator.id}")
            elif isinstance(decorator, ast.Attribute):
                parts = []
                curr = decorator
                while isinstance(curr, ast.Attribute):
                    parts.append(curr.attr)
                    curr = curr.value
                if isinstance(curr, ast.Name):
                    parts.append(curr.id)
                decorators.append(f"@{'.'.join(reversed(parts))}")
            elif isinstance(decorator, ast.Call):
                func = decorator.func
                func_name = ""
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    parts = []
                    curr = func
                    while isinstance(curr, ast.Attribute):
                        parts.append(curr.attr)
                        curr = curr.value
                    if isinstance(curr, ast.Name):
                        parts.append(curr.id)
                    func_name = ".".join(reversed(parts))
                
                args = []
                for arg in decorator.args:
                    if isinstance(arg, ast.Constant):
                        args.append(repr(arg.value))
                args_str = ", ".join(args)
                decorators.append(f"@{func_name}({args_str})")
        return decorators

    @staticmethod
    def get_function_signature(node: ast.FunctionDef, decorators: List[str]) -> str:
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        sig = f"def {node.name}({', '.join(args)}):"
        if decorators:
            return "\n".join(decorators) + "\n    " + sig
        return sig

    @staticmethod
    def parse_file(file_path: str, module_name: str, relative_path: str) -> List[Dict[str, Any]]:
        results = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content)
        except Exception as e:
            logger.warning(f"Skipping {file_path}: {e}")
            return []

        file_dir = os.path.dirname(relative_path)
        file_name = os.path.basename(relative_path)

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                model_name = ""
                
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                if target.id == "_name" and isinstance(item.value, ast.Constant):
                                    model_name = item.value.value
                                elif target.id == "_inherit" and not model_name:
                                    if isinstance(item.value, ast.Constant):
                                        model_name = item.value.value
                                    elif isinstance(item.value, ast.List) and item.value.elts:
                                        if isinstance(item.value.elts[0], ast.Constant):
                                            model_name = item.value.elts[0].value

                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        decorators = OdooASTParser.get_decorators(item)
                        func_signature = OdooASTParser.get_function_signature(item, decorators)
                        docstring = (ast.get_docstring(item) or "").strip()
                        
                        try:
                            source_segment = ast.get_source_segment(content, item)
                        except:
                            source_segment = None
                        
                        results.append({
                            "Module Name": module_name,
                            "File Directory": file_dir,
                            "File Name": file_name,
                            "model name": model_name,
                            "Class name": class_name,
                            "Function name": func_signature,
                            "Description": docstring,
                            "Source Code": source_segment
                        })
        return results

# --- Main Logic ---

class DatasetGenerator:
    def __init__(self, odoo_path: str, output_file: str, provider: Optional[LLMProvider] = None, 
                 concurrency: int = 5, limit: int = 0, timeout_hours: float = 0):
        self.odoo_path = odoo_path
        self.output_file = output_file
        self.provider = provider
        self.concurrency = concurrency
        self.limit = limit
        self.timeout_hours = timeout_hours
        self.start_time = time.time()
        self.checkpoint_file = f"{output_file}.checkpoint.csv"
        self.fieldnames = ["Module Name", "File Directory", "File Name", "model name", "Class name", "Function name", "Description"]
        self.processed_keys: Set[Tuple[str, str, str]] = set()

    def _is_timeout(self) -> bool:
        if self.timeout_hours <= 0:
            return False
        elapsed = (time.time() - self.start_time) / 3600
        return elapsed >= self.timeout_hours

    def _get_key(self, record: Dict[str, Any]) -> Tuple[str, str, str]:
        return (str(record['Module Name']), str(record['File Name']), str(record['Function name']))

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            try:
                df = pd.read_csv(self.checkpoint_file)
                for _, row in df.iterrows():
                    key = (str(row['Module Name']), str(row['File Name']), str(row['Function name']))
                    self.processed_keys.add(key)
                logger.info(f"Loaded {len(self.processed_keys)} records from checkpoint.")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")

    def scan_odoo(self) -> List[Dict[str, Any]]:
        addons_path = os.path.join(self.odoo_path, "addons")
        if not os.path.exists(addons_path):
            addons_path = self.odoo_path

        logger.info(f"Scanning {addons_path} for Odoo modules...")
        tasks = []
        modules = [d for d in os.listdir(addons_path) if os.path.isdir(os.path.join(addons_path, d))]
        modules.sort()

        for module in tqdm(modules, desc="Scanning Modules"):
            module_path = os.path.join(addons_path, module)
            if not os.path.exists(os.path.join(module_path, "__init__.py")) and \
               not os.path.exists(os.path.join(module_path, "__manifest__.py")):
                continue

            for root, _, files in os.walk(module_path):
                for file in files:
                    if file.endswith(".py"):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, module_path)
                        tasks.extend(OdooASTParser.parse_file(full_path, module, rel_path))
        
        logger.info(f"Found {len(tasks)} functions in total.")
        return tasks

    def process_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = self._get_key(record)
        if key in self.processed_keys:
            return None

        if self.provider and record.get("Source Code"):
            try:
                new_desc = self.provider.generate_description(record["Source Code"])
                if new_desc:
                    record["Description"] = new_desc.replace('\n', ' ').replace('\r', '')
            except Exception as e:
                logger.error(f"AI Generation failed for {record['Function name']}: {e}")
                # Keep original docstring on failure

        # Clean record for export
        export_record = {k: record[k] for k in self.fieldnames}
        # Sanitize for Excel compatibility
        if export_record["Description"]:
            export_record["Description"] = "".join(c for c in str(export_record["Description"]) 
                                                  if (0x20 <= ord(c) <= 0xD7FF) or c in ('\t', '\n', '\r'))
        return export_record

    def run(self):
        self.load_checkpoint()
        all_tasks = self.scan_odoo()
        
        # Filter tasks that haven't been processed
        pending_tasks = [t for t in all_tasks if self._get_key(t) not in self.processed_keys]
        
        if self.limit > 0:
            pending_tasks = pending_tasks[:self.limit]
            
        logger.info(f"Processing {len(pending_tasks)} pending tasks...")

        # Initialize CSV
        write_header = not os.path.exists(self.checkpoint_file) or os.stat(self.checkpoint_file).st_size == 0
        
        with open(self.checkpoint_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = {executor.submit(self.process_record, task): task for task in pending_tasks}
                
                pbar = tqdm(total=len(pending_tasks), desc="Generating Descriptions")
                try:
                    for future in concurrent.futures.as_completed(futures):
                        if self._is_timeout():
                            logger.info(f"Graceful timeout reached ({self.timeout_hours}h). Stopping and saving...")
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            break
                            
                        result = future.result()
                        if result:
                            writer.writerow(result)
                            csvfile.flush()
                            self.processed_keys.add(self._get_key(result))
                        pbar.update(1)
                except KeyboardInterrupt:
                    logger.info("Interrupted by user. Saving progress...")
                finally:
                    pbar.close()

        self.export_to_excel()

    def export_to_excel(self):
        logger.info("Converting checkpoint to final Excel file...")
        if not os.path.exists(self.checkpoint_file):
            logger.error("Checkpoint file missing.")
            return

        try:
            df = pd.read_csv(self.checkpoint_file)
            if df.empty:
                logger.warning("No data to export.")
                return

            with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
                groups = df.groupby("Module Name")
                sheet_names: Set[str] = set()
                
                for module_name, group_df in groups:
                    clean_name = "".join(c for c in str(module_name) if c not in ":\\/?*[]")
                    sheet_name = clean_name[:31]
                    
                    # Handle duplicate sheet names after truncation
                    base_sheet_name = sheet_name
                    counter = 1
                    while sheet_name in sheet_names:
                        suffix = f"_{counter}"
                        sheet_name = base_sheet_name[:31-len(suffix)] + suffix
                        counter += 1
                    
                    sheet_names.add(sheet_name)
                    group_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logger.info(f"SUCCESS: Dataset saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to export to Excel: {e}")

# --- CLI Entrypoint ---

def main():
    parser = argparse.ArgumentParser(description="Generate Odoo Dataset with AI descriptions")
    parser.add_argument("odoo_path", help="Path to Odoo source code")
    parser.add_argument("output_file", help="Output Excel file path")
    
    # Provider Config
    parser.add_argument("--provider", choices=['ollama', 'openai'], default='ollama', help="AI Provider")
    parser.add_argument("--model", help="Model name (e.g., llama3, gpt-4o)")
    parser.add_argument("--ollama-model", help="Alias for --model (backward compatibility)")
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"), help="Ollama Base URL")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API Key")
    parser.add_argument("--openai-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="OpenAI Base URL")
    
    # Performance
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent AI requests")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of functions to process")
    parser.add_argument("--timeout", type=float, default=0, help="Stop after X hours and export what we have (useful for Kaggle/Colab)")
    
    args = parser.parse_args()

    # Handle backward compatibility for --ollama-model
    if args.ollama_model and not args.model:
        args.model = args.ollama_model

    # Determine Provider
    provider = None
    if args.model:
        if args.provider == 'ollama':
            provider = OllamaProvider(model=args.model, base_url=args.ollama_url)
        elif args.provider == 'openai':
            if not args.openai_key:
                logger.error("OpenAI API Key is required for openai provider.")
                sys.exit(1)
            provider = OpenAIProvider(model=args.model, api_key=args.openai_key, base_url=args.openai_url)

    # Run Generator
    generator = DatasetGenerator(
        odoo_path=args.odoo_path,
        output_file=args.output_file,
        provider=provider,
        concurrency=args.concurrency,
        limit=args.limit,
        timeout_hours=args.timeout
    )
    generator.run()

if __name__ == "__main__":
    main()
