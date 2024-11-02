import pandas as pd
import numpy as np
import requests
import io
import os
import regex as re
from tabulate import tabulate
import logging
import json
from src.logger import CustomLogger, create_log_path

# path to save the log files
log_file_path = create_log_path("process_dataset")
# create the custom logger object
logger = CustomLogger(logger_name="process_dataset", log_filename=log_file_path)
# set the level of logging to INFO
logger.set_log_level(level=logging.INFO)

class DataLoader:
    """Class to handle data loading operations"""
    
    def __init__(self, file_path, chunk_size=10000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.df_raw = None

    def load_data(self):
        """Load data from JSON file using chunks"""
        try:
            chunks = []
            for chunk in pd.read_json(self.file_path, lines=True, chunksize=self.chunk_size):
                chunks.append(chunk)
            self.df_raw = pd.concat(chunks, ignore_index=True)
            logger.save_logs(f"Successfully loaded data from {self.file_path}", log_level='info')
            return self.df_raw
        except Exception as e:
            logger.save_logs(f"Error loading data: {str(e)}", log_level='error')
            raise

class DataProcessor:
    """Class to handle data processing operations"""
    
    def __init__(self, dataframe):
        self.df = dataframe
        self.metrics = {}

    def process_data(self):
        """Process the loaded data"""
        try:
            # Add your data processing logic here
            
            # Record metrics
            self.metrics['total_rows'] = len(self.df)
            self.metrics['columns'] = list(self.df.columns)
            
            logger.save_logs("Data processing completed successfully", log_level='info')
            return self.df
        except Exception as e:
            logger.save_logs(f"Error processing data: {str(e)}", log_level='error')
            raise

    def save_processed_data(self, output_path):
        """Save processed data to CSV"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.df.to_csv(output_path, index=False)
            logger.save_logs(f"Processed data saved to {output_path}", log_level='info')
        except Exception as e:
            logger.save_logs(f"Error saving processed data: {str(e)}", log_level='error')
            raise

class DataAnalyzer:
    """Class to handle data analysis operations"""
    
    def __init__(self, dataframe):
        self.df = dataframe
        self.metrics = {}

    def analyze_data(self):
        """Analyze the processed data"""
        try:
            # Add your data analysis logic here
            
            # Record analysis metrics
            self.metrics['null_counts'] = self.df.isnull().sum().to_dict()
            self.metrics['data_types'] = self.df.dtypes.astype(str).to_dict()
            
            logger.save_logs("Data analysis completed successfully", log_level='info')
            return self.df
        except Exception as e:
            logger.save_logs(f"Error analyzing data: {str(e)}", log_level='error')
            raise

def save_metrics(processor_metrics, analyzer_metrics, output_path):
    """Save metrics to JSON file"""
    try:
        metrics = {
            'processing_metrics': processor_metrics,
            'analysis_metrics': analyzer_metrics
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.save_logs(f"Metrics saved to {output_path}", log_level='info')
    except Exception as e:
        logger.save_logs(f"Error saving metrics: {str(e)}", log_level='error')
        raise

def main():
    """Main function to orchestrate the data processing pipeline"""
    try:
        # Initialize data loader
        data_loader = DataLoader('data/raw/extracted/transactions.txt')
        
        # Load the data
        df_raw = data_loader.load_data()
        
        # Process the data
        processor = DataProcessor(df_raw)
        processed_df = processor.process_data()
        
        # Save processed data
        processor.save_processed_data('data/inprogress/inprogress_transactions.csv')
        
        # Analyze the data
        analyzer = DataAnalyzer(processed_df)
        analyzed_df = analyzer.analyze_data()
        
        # Save metrics
        save_metrics(
            processor_metrics=processor.metrics,
            analyzer_metrics=analyzer.metrics,
            output_path='reports/data_processing_metrics.json'
        )
        
        logger.save_logs("Main processing pipeline completed successfully", log_level='info')
        return analyzed_df
        
    except Exception as e:
        logger.save_logs(f"Error in main processing pipeline: {str(e)}", log_level='error')
        raise

if __name__ == "__main__":
    main()