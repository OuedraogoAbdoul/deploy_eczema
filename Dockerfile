FROM python:3.8

WORKDIR /usr/src/ml_app/

ADD . /usr/src/ml_app/

RUN pip install --upgrade pip
RUN  python3.8 -m pip install --upgrade pip

RUN pip install -r /usr/src/ml_app/regression_model/requirement.txt
RUN pip install notebook

EXPOSE 5000

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root", "bash"]
