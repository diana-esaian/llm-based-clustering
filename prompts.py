topic_modelling_prompt = """You are a customer service expert identifying key concerns in customer messages. Extract a concise topic for each message and return a Python dictionary where:  

- **Keys**: Message numbers ('1', '2', '3', etc.).  
- **Values**: A concise keyword or phrase representing the main topic of concern.  

**Output Format:**  
{"1": "topic", "2": "topic", ...}  

**Constraints:**  
- Keep topics concise yet informative.  
- Ensure accuracy in capturing customer concerns.  
- Output only the dictionary—no extra text.  

**Example:**  
Input:  
1: "My order #12345 hasn't arrived yet."  
2: "The product I received was damaged during shipping. I want to return it."  
Output:
{"1": "delivery delay", "2": "return request"}  

Now, provide the customer messages."""


clustering_prompt = """You are an expert in topic clustering. Given the following list of subtopics, group them into meaningful clusters.  

Your output should be a Python dictionary where keys are cluster names (strings) and values are lists of subtopic (strings) belonging to that cluster. 

- Ensure each subtopic appears in exactly one cluster.
- Ensure that all the subtopics are assigned.  
- Ensure each subtopic is written in the same way as it appears in the input list.
- Ensure that each cluster is about one subject of customer concern.

**Desired Output (Python dictionary):**
{"ClusterName1": ["subtopicA", "subtopicB", ...], "ClusterName2": ["subtopicC", "subtopicD", ...], ...}

Your output should not include any additional symbols or text. It should only be the Python dictionary.
"""


instruction_merge = """Given a list of data clusters, identify any clusters that are semantically the same. Return a JSON object where each key represents a new, merged cluster name, and the corresponding value is a list of the original cluster names that should be merged into it. 
**Output Format:**

{"NewClusterName": ["cluster1", "cluster2", ...], "NewClusterName2": ["cluster4", "cluster6", ...]}, or {} if no mergeable clusters are found.

If some cluster is no merged, it is a should also be included in the output as both the key and the value: "SameName": ["SameName"]"""


prompt_description = """Given a list of question topics representing a cluster, generate a short concise and descriptive one-sentence explanation of the type of questions within that cluster. Return your response in the following JSON format: {"name": "<cluster_name>", "description": "<short_description>"}"""
