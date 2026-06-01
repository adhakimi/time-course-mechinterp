import argparse
import pickle
import os
import pprint

def calculate_accuracy(data_entries, relation_name):
    relation_results = {
        relation_name: {
            "total": 0,
            "top_1_correct": 0,
            "top_10_correct": 0
        }
    }
    overall_results = {
        "total": 0,
        "top_1_correct": 0,
        "top_10_correct": 0
    }
    
    for entry in data_entries:
        answer_token_idx = entry['answer_token_span'][0] - 1
        logit_lens = entry['logit_lens_result']["resid"]["final_post"]
        relevant_logit = logit_lens[answer_token_idx]

        top_1_correct = relevant_logit['rank_answer'] == 0
        top_10_correct = relevant_logit['rank_answer'] < 10

        overall_results['total'] += 1
        if top_1_correct:
            overall_results['top_1_correct'] += 1
        if top_10_correct:
            overall_results['top_10_correct'] += 1
        
        relation_results[relation_name]['total'] += 1
        if top_1_correct:
            relation_results[relation_name]['top_1_correct'] += 1
        if top_10_correct:
            relation_results[relation_name]['top_10_correct'] += 1

    micro_avg_top_1 = relation_results[relation_name]['top_1_correct'] / relation_results[relation_name]['total']
    micro_avg_top_10 = relation_results[relation_name]['top_10_correct'] / relation_results[relation_name]['total']

    return {
        "overall_results": overall_results,
        "relation_results": relation_results,
        "micro_avg_top_1": micro_avg_top_1,
        "micro_avg_top_10": micro_avg_top_10
    }

def process_all_relations(directory):
    all_relations_results = []
    total_overall = {
        "total": 0,
        "top_1_correct": 0,
        "top_10_correct": 0
    }
    relation_accuracies = {}
    
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            file_path = os.path.join(directory, filename)
            relation_name = os.path.splitext(filename)[0]
            
            with open(file_path, 'rb') as file:
                data_entries = pickle.load(file)
            
            relation_result = calculate_accuracy(data_entries, relation_name)
            all_relations_results.append(relation_result)
            
            total_overall['total'] += relation_result['overall_results']['total']
            total_overall['top_1_correct'] += relation_result['overall_results']['top_1_correct']
            total_overall['top_10_correct'] += relation_result['overall_results']['top_10_correct']
            
            relation_accuracies[relation_name] = {
                "top_1_accuracy": relation_result['micro_avg_top_1'],
                "top_10_accuracy": relation_result['micro_avg_top_10']
            }

    micro_avg_top_1 = total_overall['top_1_correct'] / total_overall['total']
    micro_avg_top_10 = total_overall['top_10_correct'] / total_overall['total']
    
    macro_avg_top_1 = sum(relation_accuracies[rel]['top_1_accuracy'] for rel in relation_accuracies) / len(relation_accuracies)
    macro_avg_top_10 = sum(relation_accuracies[rel]['top_10_accuracy'] for rel in relation_accuracies) / len(relation_accuracies)
    
    return {
        "all_relations_results": all_relations_results,
        "overall_micro_avg_top_1": micro_avg_top_1,
        "overall_micro_avg_top_10": micro_avg_top_10,
        "overall_macro_avg_top_1": macro_avg_top_1,
        "overall_macro_avg_top_10": macro_avg_top_10,
        "relation_accuracies": relation_accuracies
    }

def process_single_relation(file_path):
    relation_name = os.path.splitext(os.path.basename(file_path))[0]

    with open(file_path, 'rb') as file:
        data_entries = pickle.load(file)
    
    relation_result = calculate_accuracy(data_entries, relation_name)
    #pprint.pprint(relation_result)
    return relation_result

def main():
    #python metrics.py --output_path "/nfs/datz/olmo_models/outputs/" --revision main (--relation person_university)
    parser = argparse.ArgumentParser(description="Calculate accuracy for specific or all relations")
    parser.add_argument('--relation', type=str, required=False, help='Specific relation from Category')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for saved pickle file')
    parser.add_argument('--revision', type=str, required=True, help='Model revision to use')

    args = parser.parse_args()

    # Define directory to use
    directory = os.path.join(args.output_path, args.revision)

    if args.relation:
        # Process a single relation
        file_path = os.path.join(directory, f"{args.relation}.pkl")
        if os.path.exists(file_path):
            result = process_single_relation(file_path)
            pprint.pprint(result)
        else:
            print(f"File for relation {args.relation} does not exist.")
    else:
        # Process all relations
        if os.path.exists(directory):
            result = process_all_relations(directory)
            pprint.pprint(result)
        else:
            print(f"Directory {directory} does not exist.")

if __name__ == "__main__":
    main()
