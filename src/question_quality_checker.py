import logging
import os

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Configure logger
logger = logging.getLogger(__name__)

# Download nltk data if not already present
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")


class QuestionQualityChecker:
    def __init__(
        self,
        input_csv: str,
        question_column: str = "synthetic_questions",
        output_path: str = None,
    ):
        self.df = pd.read_csv(input_csv)
        self.question_column = question_column
        self.output_path = output_path

    def find_duplicates(self) -> pd.DataFrame:
        """Find duplicate questions in the generated questions CSV."""
        duplicates = self.df[
            self.df.duplicated(subset=[self.question_column], keep=False)
        ]
        logger.info(f"Found {len(duplicates)} duplicate questions.")
        return duplicates

    def find_similar_questions(self, similarity_threshold: float = 0.8) -> pd.DataFrame:
        """Find similar questions in the generated questions CSV based on a similarity threshold."""

        # Filter out NaN/empty values before processing
        df_clean = self.df[self.df[self.question_column].notna()].copy()
        df_clean = df_clean[
            df_clean[self.question_column].astype(str).str.strip() != ""
        ]

        if len(df_clean) == 0:
            logger.warning("No valid questions to check for similarity.")
            return pd.DataFrame()

        questions = df_clean[self.question_column].astype(str).tolist()

        # Only run similarity check if we have at least 2 questions
        if len(questions) < 2:
            logger.info("Less than 2 valid questions to compare for similarity.")
            return pd.DataFrame()

        vectorizer = TfidfVectorizer().fit_transform(questions)
        vectors = vectorizer.toarray()
        cosine_matrix = cosine_similarity(vectors)

        similar_pairs = []
        for i in range(len(cosine_matrix)):
            for j in range(i + 1, len(cosine_matrix)):
                if cosine_matrix[i][j] >= similarity_threshold:
                    similar_pairs.append(
                        (
                            df_clean.iloc[i].get("id", i),
                            df_clean.iloc[j].get("id", j),
                            cosine_matrix[i][j],
                        )
                    )

        similar_df = pd.DataFrame(
            similar_pairs,
            columns=["Question_ID_1", "Question_ID_2", "Similarity_Score"],
        )
        logger.info(
            f"Found {len(similar_df)} similar question pairs with similarity >= {similarity_threshold}."
        )
        return similar_df

    def find_length_outliers(
        self, min_length: int = 5, max_length: int = 50
    ) -> pd.DataFrame:
        """Find questions that are too short or too long based on word count thresholds."""

        self.df["word_count"] = self.df[self.question_column].apply(
            lambda x: len(str(x).split())
        )
        outliers = self.df[
            (self.df["word_count"] < min_length) | (self.df["word_count"] > max_length)
        ]
        logger.info(
            f"Found {len(outliers)} questions outside the length range of {min_length}-{max_length} words."
        )
        return outliers

    # find questions with low lexical diversity (type-token ratio below a threshold)
    def find_low_lexical_diversity(
        self, diversity_threshold: float = 0.5
    ) -> pd.DataFrame:
        """Find questions with low lexical diversity (type-token ratio below a threshold)."""

        def type_token_ratio(text):
            tokens = str(text).split()
            types = set(tokens)
            return len(types) / len(tokens) if len(tokens) > 0 else 0

        self.df["type_token_ratio"] = self.df[self.question_column].apply(
            type_token_ratio
        )
        low_diversity = self.df[self.df["type_token_ratio"] < diversity_threshold]
        logger.info(
            f"Found {len(low_diversity)} questions with type-token ratio below {diversity_threshold}."
        )
        return low_diversity

    # find questions with common stopwords ratio above a threshold
    def find_high_stopword_ratio(
        self, stopword_ratio_threshold: float = 0.5
    ) -> pd.DataFrame:
        """Find questions with high stopword ratio above a threshold."""

        stop_words = set(stopwords.words("english"))

        def stopword_ratio(text):
            tokens = str(text).split()
            if len(tokens) == 0:
                return 0
            stopword_count = sum(1 for token in tokens if token.lower() in stop_words)
            return stopword_count / len(tokens)

        self.df["stopword_ratio"] = self.df[self.question_column].apply(stopword_ratio)
        high_stopword = self.df[self.df["stopword_ratio"] > stopword_ratio_threshold]
        logger.info(
            f"Found {len(high_stopword)} questions with stopword ratio above {stopword_ratio_threshold}."
        )
        return high_stopword

    # run all quality checks and return a summary report
    def run_all_checks(self) -> dict:
        """Run all quality checks and return a summary report."""
        report = {}
        report["duplicates"] = self.find_duplicates()
        report["similar_questions"] = self.find_similar_questions()
        report["length_outliers"] = self.find_length_outliers()
        report["low_lexical_diversity"] = self.find_low_lexical_diversity()
        report["high_stopword_ratio"] = self.find_high_stopword_ratio()

        # Log summary
        logger.info("Quality Check Summary:")
        for check_name, check_result in report.items():
            if isinstance(check_result, dict):
                logger.info(f"  {check_name}: {check_result}")
            else:
                logger.info(f"  {check_name}: {len(check_result)} issues found")

        # output report to CSV files if output_path is provided
        if self.output_path:
            for check_name, check_result in report.items():
                if not check_result.empty:
                    out_file = os.path.join(
                        self.output_path, f"{check_name}_report.csv"
                    )
                    # output_file = f"{self.output_path}_{check_name}.csv"
                    check_result.to_csv(out_file, index=False)
                    logger.info(f"  {check_name} results saved to {out_file}")
        return report
