FROM duanegoodner/dev_envs:postgres

ARG LOCAL_MIMICIII_BUILD_TOOLS
ARG POSTGRES_ENTRYPOINT

USER root

# RUN mkdir -p /mimic-code /mimic_data \
#  && cd /mimic-code \
#  # clone the postgres build scripts into a local folder
#  && git init \
#  && git remote add -f origin https://github.com/MIT-lcp/mimic-code \
#  && git config core.sparseCheckout true \
#  && echo "mimic-iii/buildmimic/postgres/" >> .git/info/sparse-checkout \
#  && echo "mimic-iii/buildmimic/docker/"   >> .git/info/sparse-checkout \
#  && git pull origin main \
#  # copy the build scripts into a different folder and remove the temp folder
#  && cp -r mimic-iii/buildmimic /docker-entrypoint-initdb.d/ \
#  && cp mimic-iii/buildmimic/docker/setup.sh /docker-entrypoint-initdb.d/ \
#  && rm -rf /mimic-code

COPY ${LOCAL_MIMICIII_BUILD_TOOLS}/docker/setup.sh \
  ${POSTGRES_ENTRYPOINT}/setup.sh
COPY ${LOCAL_MIMICIII_BUILD_TOOLS}/docker \
  ${POSTGRES_ENTRYPOINT}/buildmimic/docker
COPY ${LOCAL_MIMICIII_BUILD_TOOLS}/postgres \
  ${POSTGRES_ENTRYPOINT}/buildmimic/postgres

COPY ./src/docker_db/pg_entrypoint.sh ${POSTGRES_ENTRYPOINT}/pg_entrypoint.sh