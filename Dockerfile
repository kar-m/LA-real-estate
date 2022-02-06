FROM python:3.10.2-slim-buster
RUN mkdir /karen-plot
WORKDIR /karen-plot
COPY ./ /karen-plot/
RUN pip install -r requirements.txt 	
EXPOSE 8050
CMD [ "python3", "real_estate_dash.py" ]