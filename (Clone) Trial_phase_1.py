# Databricks notebook source
# Create a text widget
dbutils.widgets.text("name", "")

# Access the value of the text widget
name = dbutils.widgets.get("name")

# Print the name
print(name)

