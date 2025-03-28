import os
import json
import asyncio
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv
from collections import defaultdict

from prompts import *

load_dotenv()

class LLMClustering():
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.dataframe = None
        self.request_count = 0  
    
    def chunk_messages(self, messages, batch_size=300):
        """
        Split messages into smaller batches for processing.
        
        :param messages: List of messages.
        :param batch_size: Number of messages per batch.
        :return: List of message batches.
        """
        return [messages[i:i + batch_size] for i in range(0, len(messages), batch_size)]

    def preprocess(self, data_path):
        """
        Load and preprocess the dataset from a CSV file.
        
        :param data_path: Path to the CSV file.
        :return: List of message batches.
        """
        self.dataframe = pd.read_csv(data_path)[['instruction', 'category', 'intent']]
        all_questions = self.dataframe['instruction'].values.tolist()
        formatted_messages = "\n".join(f"{i+1}. {msg}" for i, msg in enumerate(all_questions))
        batches = self.chunk_messages(formatted_messages.split('\n'))
        return batches
    
    async def extract_topic_async(self, batch, instruction, semaphore):
        """
        Asynchronously extracts topic of concern for a batch of messages.
        
        :param batch: List of messages.
        :param instruction: System instruction for the model.
        :param semaphore: Async semaphore to limit concurrent requests.
        :return: Generated response text or None if an error occurs.
        """
        async with semaphore:  
            try:
                response = await self.client.aio.models.generate_content(  
                    model='gemini-2.0-flash',
                    contents=batch,
                    config=types.GenerateContentConfig(
                        system_instruction=instruction
                    ),
                )
                self.request_count += 1
                if self.request_count % 10 == 0:  
                    await asyncio.sleep(20)  
                return response.text
            except asyncio.TimeoutError:
                print("Request timed out")
                return None
            except Exception as e:
                print(f"Error: {e}")
                return None

    async def run_topic_extract(self, batches):
        """
        Process all batches asynchronously.
        
        :param batches: List of message batches.
        :return: List of generated responses.
        """
        semaphore = asyncio.Semaphore(3)  
        tasks = [self.extract_topic_async(batch, topic_modelling_prompt, semaphore) for batch in batches]
        results = await asyncio.gather(*tasks)
        return results
    
    def postprocessing(self, results, eval_list=True):
        """
        Clean and process model responses.
        
        :param results: List of raw responses.
        :param eval_list: Whether to evaluate JSON-like strings.
        :return: Processed responses.
        """
        cleaned = [result.replace("python", "").replace("\n", " ").replace("```", "").replace("json", "") for result in results if result is not None]
        if eval_list:
            cleaned = [eval(result) for result in cleaned]  
        return cleaned
    
    def pre_clustering(self, list_json_results):
        """
        Extract unique issues and prepare them for clustering.
        
        :param list_json_results: List of JSON results from topic modeling.
        :return: List of topic batches.
        """
        combined = {k: v for d in list_json_results for k, v in d.items()}
        unique_issues = set(combined.values())
        batches_topics = self.chunk_messages(list(unique_issues), batch_size=300)
        return batches_topics

    async def cluster_content_async(self, batch, clustering_prompt, semaphore):
        """
        Asynchronously cluster content based on topics.
        
        :param batch: List of topics.
        :param clustering_prompt: System instruction for clustering.
        :param semaphore: Async semaphore to limit concurrent requests.
        :return: Clustered response text or None if an error occurs.
        """
        async with semaphore:  
            try:
                response = await self.client.aio.models.generate_content(  
                    model='gemini-1.5-pro',
                    contents=batch,
                    config=types.GenerateContentConfig(
                        system_instruction=clustering_prompt
                    ),
                )
                self.request_count += 1
                if self.request_count % 10 == 0:  
                    await asyncio.sleep(20)  
                return response.text
            except asyncio.TimeoutError:
                print("Request timed out")
                return None
            except Exception as e:
                print(f"Error: {e}")
                return None

    async def run_clustering(self, batches_topics):
        """
        Process clustering asynchronously.
        
        :param batches_topics: List of topic batches.
        :return: List of clustered responses.
        """
        semaphore = asyncio.Semaphore(2)  
        tasks = [self.cluster_content_async(batch, clustering_prompt, semaphore) for batch in batches_topics]
        results = await asyncio.gather(*tasks)
        print('clustering')
        return results

    def merge_dicts(self, *dicts):
        """
        Merge multiple dictionaries by combining their list values.
        
        :param dicts: Dictionaries to merge.
        :return: A merged dictionary with combined lists.
        """        
        merged = defaultdict(list)
        for d in dicts:
            for key, value in d.items():
                merged[key].extend(value)  
        return dict(merged)
    
    def merge(self, list_clusters):
        """
        Merge cluster dictionaries and generate a final set of clusters using the model.
        
        :param list_clusters: List of dictionaries containing clustered topics.
        :return: Merged clusters and main topic clusters.
        """
        merged_list_clusters = self.merge_dicts(*list_clusters)
        response_merge_2 = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=list(merged_list_clusters.keys()),
            config=types.GenerateContentConfig(
                system_instruction = instruction_merge
                    ),
        )
        main_clusters = eval(self.postprocessing([response_merge_2.text], eval_list=False)[0])
        print('merged')
        return merged_list_clusters, main_clusters
        
        
    def label(self, merged_list_clusters, main_clusters):
        """
        Assign labels to clustered topics and update the dataframe.
        
        :param merged_list_clusters: Dictionary of merged clusters.
        :param main_clusters: Dictionary of main topic clusters.
        :return: Dictionary mapping clusters to topics.
        """
        updated_key_dict = {
            key: [val for sub in subkeys if sub in merged_list_clusters for val in merged_list_clusters[sub]]
            for key, subkeys in main_clusters.items()
        }
        
        topic_to_cluster = {topic: cluster for cluster, topics in updated_key_dict.items() for topic in topics}
        self.dataframe["topic"] = self.dataframe["instruction"].apply(lambda x: next((k for k, v in merged_list_clusters.items() if x in v), None))
        self.dataframe["cluster"] = self.dataframe["topic"].map(topic_to_cluster)
        print('labeled')
        return updated_key_dict

    
    async def summary_content_async(self, cluster, prompt_description, semaphore):
        """
        Generate a summary of clustered topics asynchronously.
        
        :param cluster: Cluster to summarize.
        :param prompt_description: System instruction for summarization.
        :param semaphore: Async semaphore to limit concurrent requests.
        :return: Generated summary text or None if an error occurs.
        """
        async with semaphore:  
            try:
                response = await self.client.aio.models.generate_content(  
                    model='gemini-2.0-flash',
                    contents=str(cluster),
                    config=types.GenerateContentConfig(
                        system_instruction=prompt_description
                    ),
                )
                self.request_count += 1
                if self.request_count % 10 == 0:  
                    await asyncio.sleep(20)  
                return response.text
            except asyncio.TimeoutError:
                print("Request timed out")
                return None
            except Exception as e:
                print(f"Error: {e}")
                return None

    async def run_summary(self, cluster_for_summary):
        semaphore = asyncio.Semaphore(2)  
        tasks = [self.summary_content_async(cluster, prompt_description, semaphore) for cluster in cluster_for_summary]
            
        results = await asyncio.gather(*tasks)
        clean_summaries = self.postprocessing(results)
        print('summary')
        return clean_summaries

        
    async def run(self, data_path):
        """
        Execute the entire clustering process asynchronously.
        
        :param data_path: Path to the input CSV file.
        """
        batches = self.preprocess(data_path)
        results = await self.run_topic_extract(batches)  
        list_json_results = self.postprocessing(results)
        batches_topics = self.pre_clustering(list_json_results)
        clusters = await self.run_clustering(batches_topics)
        list_clusters = self.postprocessing(clusters)
        merged_list_clusters, main_clusters = self.merge(list_clusters)
        updated_key_dict =self.label(merged_list_clusters, main_clusters)
        cluster_for_summary = [{"name": cluster, "topics": topics} for cluster, topics in updated_key_dict.items()]
        output = await self.run_summary(cluster_for_summary)  
        counts = self.dataframe["cluster"].value_counts()
        final_output = [{**result, 'count': counts.get(result['name'], 0)} for result in output]
        with open('output.json', 'w') as f:
            json.dump(final_output, f, default=int) 
            

llm_clustering = LLMClustering(os.getenv('GEMINI_API_KEY'))
asyncio.run(llm_clustering.run('Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'))