from Bio import SeqIO
from Bio import pairwise2
import pandas as pd
from Bio.SubsMat import MatrixInfo
import multiprocessing as mp
import argparse
import os

parser = argparse.ArgumentParser(
    description='predict specific serotype of phage depolymerase')
parser.add_argument('-f', default='', type=str)
parser.add_argument('--reference_path', type=str, default='./data/subseq/published_dpos_knownKL_subseq_expand.fasta',
                    help='path to reference depolymerase file')
parser.add_argument('--query_path', type=str, default='./data/subseq/phage_proteins_positive_subseq.fasta',
                    help='path to depolymerase file to be predicted')
parser.add_argument('--output', type=str, default='./data/',
                    help='path to output file')
parser.add_argument('--k', type=int, default=1, help='return top k best alignment')
args = parser.parse_args()

input_file = args.query_path
reference = args.reference_path

# use blosum62 matrix for alignment
blosum62 = MatrixInfo.blosum62

# def align_sequences(query_record, subject_records):
#     max_score = 0
#     best_subject_desc = ""
#     best_target_ktype = ""
    
#     for subject_record in subject_records:
#         alignments = pairwise2.align.localds(query_record.seq, subject_record.seq, blosum62, -10, -0.5)
#         current_score = max([alignment.score for alignment in alignments])
        
#         if current_score > max_score:
#             max_score = current_score
#             best_subject_desc = subject_record.description
#             best_target_ktype = subject_record.description.split(" ")[1]
    
#     # calculate max possible score (align with itself)
#     max_possible_alignments = pairwise2.align.localds(query_record.seq, query_record.seq, blosum62, -10, -0.5)
#     max_possible_score = max([alignment.score for alignment in max_possible_alignments])
    
#     # calculate normalized score
#     normalized_score = max_score / max_possible_score if max_possible_score > 0 else 0
    
#     return query_record.description, best_subject_desc, max_score, normalized_score, best_target_ktype

def align_sequences(query_record, subject_records):
    alignments = []
    
    for subject_record in subject_records:
        alignment = pairwise2.align.localds(query_record.seq, subject_record.seq, blosum62, -10, -0.5)
        score = max([a.score for a in alignment])

        # calculate max possible score (align with itself)
        max_possible_alignments = pairwise2.align.localds(query_record.seq, query_record.seq, blosum62, -10, -0.5)
        max_possible_score = max([a.score for a in max_possible_alignments])

        # calculate normalized score
        normalized_score = score / max_possible_score if max_possible_score > 0 else 0
        alignments.append((query_record.description, subject_record.description, normalized_score, subject_record.description.split(" ")[1]))
    
    alignments.sort(key=lambda x: x[2], reverse=True)
    
    # select top k alignments
    top_k_alignments = alignments[:args.k]
    
    return top_k_alignments

subject_records = list(SeqIO.parse(reference, "fasta"))

results = []

# # multiprocessing
# with mp.Pool(processes=mp.cpu_count()) as pool:
#     query_records = list(SeqIO.parse(input_file, "fasta"))
#     results = pool.starmap(align_sequences, [(query_record, subject_records) for query_record in query_records])

# df = pd.DataFrame(results, columns=["query", "subject", "score", "normalized_score", "predict_Ktype"])

# df.to_csv(os.path.join(args.output, "predicted_serotype.csv"), index=False)

# multiprocessing
with mp.Pool(processes=mp.cpu_count()) as pool:
    query_records = list(SeqIO.parse(input_file, "fasta"))
    results = pool.starmap(align_sequences, [(query_record, subject_records) for query_record in query_records])

results = [result for sublist in results for result in sublist]

df = pd.DataFrame(results, columns=["query", "subject", "score", "predict_Ktype"])

df.to_csv(os.path.join(args.output, "predicted_serotype.csv"), index=False)