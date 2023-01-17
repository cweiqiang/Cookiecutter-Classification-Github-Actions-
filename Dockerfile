FROM python:3.9.7
COPY . /ML_proj
WORKDIR /ML_proj
RUN pip install -r requirements.txt
EXPOSE 8501
RUN chmod 777 /ML_proj/run_without_gpu.sh
CMD ["/bin/bash", "/ML_proj/run.sh"]
