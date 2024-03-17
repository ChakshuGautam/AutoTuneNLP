# Generated by Django 4.2.10 on 2024-03-16 15:35

import django.contrib.postgres.fields
import django.core.validators
from django.db import migrations, models
import django.db.models.deletion
import uuid
import workflow.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Dataset',
            fields=[
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('id', models.UUIDField(default=uuid.uuid4, primary_key=True, serialize=False)),
                ('huggingface_id', models.UUIDField(blank=True, null=True)),
                ('uploaded_at', models.DateTimeField(blank=True, null=True)),
                ('is_generated_at_autotune', models.BooleanField(default=False)),
                ('latest_commit_hash', models.UUIDField(blank=True, null=True)),
                ('name', models.CharField(max_length=255)),
                ('is_locally_cached', models.BooleanField(default=False)),
            ],
        ),
        migrations.CreateModel(
            name='Log',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('api_url', models.CharField(max_length=255)),
                ('model', models.CharField(max_length=255)),
                ('system', models.TextField()),
                ('user', models.TextField()),
                ('text', models.TextField()),
                ('result', models.TextField()),
                ('latency_ms', models.IntegerField(default=-1)),
            ],
        ),
        migrations.CreateModel(
            name='MLModel',
            fields=[
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('id', models.UUIDField(default=uuid.uuid4, primary_key=True, serialize=False)),
                ('huggingface_id', models.UUIDField(blank=True, null=True)),
                ('uploaded_at', models.DateTimeField(blank=True, null=True)),
                ('latest_commit_hash', models.UUIDField(blank=True, null=True)),
                ('is_trained_at_autotune', models.BooleanField(default=False)),
                ('name', models.CharField(max_length=255)),
                ('is_locally_cached', models.BooleanField(default=False)),
            ],
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('user_id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('user_name', models.CharField(max_length=255)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('huggingface_user_id', models.CharField(max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name='WorkflowConfig',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, unique=True)),
                ('system_prompt', models.TextField()),
                ('user_prompt_template', models.TextField()),
                ('json_schema', models.JSONField(blank=True, default=dict, null=True)),
                ('parameters', models.JSONField(blank=True, default=dict, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Workflows',
            fields=[
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('workflow_id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('workflow_name', models.CharField(max_length=255)),
                ('workflow_type', models.CharField(choices=[('QnA', 'QnA Example')], default='QnA', max_length=50)),
                ('tags', django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=255), size=None)),
                ('total_examples', models.IntegerField()),
                ('split', django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(100)]), default=workflow.models.default_split, size=None, validators=[workflow.models.validate_split])),
                ('llm_model', models.CharField(choices=[('gpt-4-0125-preview', 'gpt-4-0125-preview'), ('gpt-4-turbo-preview', 'gpt-4-turbo-preview'), ('gpt-4-1106-preview', 'gpt-4-1106-preview'), ('gpt-4-vision-preview', 'gpt-4-vision-preview'), ('gpt-3.5-turbo-0125', 'gpt-3.5-turbo-0125'), ('gpt-3.5-turbo', 'gpt-3.5-turbo')], max_length=255)),
                ('cost', models.IntegerField(default=0)),
                ('status', models.CharField(choices=[('SETUP', 'Setup'), ('ITERATION', 'Iteration'), ('GENERATION', 'Generation'), ('TRAINING', 'Training'), ('PUSHING_DATASET', 'Pushing Dataset'), ('PUSHING_MODEL', 'Pushing Model')], default='SETUP', max_length=20)),
                ('status_details', models.JSONField(default=dict)),
                ('dataset', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='workflow', to='workflow.dataset')),
                ('model', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='+', to='workflow.mlmodel')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='workflow', to='workflow.user')),
            ],
        ),
        migrations.CreateModel(
            name='Task',
            fields=[
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=255)),
                ('format', models.JSONField(default=dict)),
                ('status', models.CharField(default='Starting', max_length=255)),
                ('workflow', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='task', to='workflow.workflows')),
            ],
        ),
        migrations.CreateModel(
            name='Prompt',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user', models.TextField(blank=True, null=True)),
                ('source', models.TextField(blank=True, null=True)),
                ('workflow', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='prompt', to='workflow.workflows')),
            ],
        ),
        migrations.CreateModel(
            name='Examples',
            fields=[
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('example_id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('text', models.TextField()),
                ('label', models.CharField(max_length=255)),
                ('reason', models.TextField(max_length=255)),
                ('workflow', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='examples', to='workflow.workflows')),
            ],
        ),
    ]
