# Mava Edu image
FROM instadeepct/mava:sc2-latest as mava-edu
ENV PATH /mava/bin:$PATH
RUN pip install --upgrade pip setuptools wheel
# Location of mava folder
ARG folder=/home/app/mava
## working directory
WORKDIR ${folder}
# Install packages
RUN pip install -r educational/requirements.txt
EXPOSE 6006
