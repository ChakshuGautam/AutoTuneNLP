# Example: 0034_alter_mlmodel_latest_commit_hash.py

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("workflow", "0034_datasetdata_record_id_datasetdata_user"),  # Update to the previous migration filename
    ]

    operations = [
        migrations.AlterField(
            model_name="mlmodel",
            name="latest_commit_hash",
            field=models.CharField(max_length=255, null=True, blank=True),
        ),
    ]
