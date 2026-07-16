# main.py

import json

def load_evaluation_results(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def process_results(results):
    for result in results:
        print(f"Document: {result['doc_name']}")
        print(f"Question: {result['question']}")
        print(f"Expected Answer: {result['expected_answer']}")
        print(f"Generated Answer: {result['generated_answer']}")
        print(f"Overlap Score: {result['overlap_score']}\n")

def main():
    data_file = '../data/evaluation_results.json'
    evaluation_data = load_evaluation_results(data_file)
    
    print(f"Average Score: {evaluation_data['average_score']}")
    process_results(evaluation_data['results'])

if __name__ == "__main__":
    main()