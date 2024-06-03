from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from text_processing_tasks import (
    download_data_from_gcs,
    preprocess_text,
    upload_processed_text_to_gcs,
    analyze_emotions,
    upload_emotions_to_gcs,
    perform_topic_modeling,
    upload_topics_to_gcs,
    tokenize_and_pad_sequences,
    upload_padded_sequences_to_gcs
)

from airflow import configuration as conf

# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# Default arguments for the DAG
default_args = {
    'owner': 'Pragnesh Anekal',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 2),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'text_processing_pipeline',
    default_args=default_args,
    description='A simple text processing pipeline',
    schedule_interval=timedelta(days=1),
)

# Define the tasks
download_data_task = PythonOperator(
    task_id='download_data_from_gcs',
    python_callable=download_data_from_gcs,
    dag=dag,
)

preprocess_text_task = PythonOperator(
    task_id='preprocess_text',
    python_callable=preprocess_text,
    op_args=[download_data_task.output],
    dag=dag,
)

upload_processed_text_task = PythonOperator(
    task_id='upload_processed_text_to_gcs',
    python_callable=upload_processed_text_to_gcs,
    dag=dag,
)

analyze_emotions_task = PythonOperator(
    task_id='analyze_emotions',
    python_callable=analyze_emotions,
    dag=dag,
)

upload_emotions_task = PythonOperator(
    task_id='upload_emotions_to_gcs',
    python_callable=upload_emotions_to_gcs,
    dag=dag,
)

perform_topic_modeling_task = PythonOperator(
    task_id='perform_topic_modeling',
    python_callable=perform_topic_modeling,
    dag=dag,
)

upload_topics_task = PythonOperator(
    task_id='upload_topics_to_gcs',
    python_callable=upload_topics_to_gcs,
    dag=dag,
)

tokenize_and_pad_sequences_task = PythonOperator(
    task_id='tokenize_and_pad_sequences',
    python_callable=tokenize_and_pad_sequences,
    dag=dag,
)

upload_padded_sequences_task = PythonOperator(
    task_id='upload_padded_sequences_to_gcs',
    python_callable=upload_padded_sequences_to_gcs,
    dag=dag,
)

# Define task dependencies
download_data_task >> preprocess_text_task >> upload_processed_text_task
preprocess_text_task >> analyze_emotions_task >> upload_emotions_task
preprocess_text_task >> perform_topic_modeling_task >> upload_topics_task
preprocess_text_task >> tokenize_and_pad_sequences_task >> upload_padded_sequences_task