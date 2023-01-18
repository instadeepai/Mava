# Mava Edu image
FROM instadeepct/mava:sc2-latest as mava-edu
ENV PATH /mava/bin:$PATH
RUN pip install --upgrade pip setuptools wheel
# Location of mava folder
ARG folder=/home/app/mava/educational
## working directory
WORKDIR ${folder}
## Copy code from current path
COPY . ${folder}
# Install packages
RUN pip install -r educational/requirements.txt
EXPOSE 6006
