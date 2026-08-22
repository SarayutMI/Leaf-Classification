// Leaf-Classification CI/CD — same shape as the yamwisdom pipeline on this
// host (Vault -> .env -> compose build -> test -> deploy -> health check).
//
// One pipeline serves both environments; the branch selects everything else:
//
//   develop -> leaf-api-dev   docker-compose.dev.yml    API :15870
//   main    -> leaf-api       docker-compose.prod.yml   API :16870
//
// Any other branch builds and tests but never deploys.
//
// IMPORTANT — do not add a stage that bind-mounts $WORKSPACE into a container
// running as root. That is what left root-owned files behind in the coop and
// lookdata jobs, which cleanWs() then could not delete. Every stage below
// either runs inside an already-built image or mounts nothing.

pipeline {
    agent any

    environment {
        VAULT_ADDR    = "https://Vault:8200"
        VAULT_RESOLVE = "Vault:8200:127.0.0.1"
        VAULT_CACERT  = credentials('VAULT_CACERT')
        VAULT_MOUNT   = "yamwisdom"
        VAULT_SECRET  = "ml_api"

        // BuildKit powers the --mount=type=cache line in the Dockerfile.
        DOCKER_BUILDKIT          = "1"
        COMPOSE_DOCKER_CLI_BUILD = "1"

        REGISTRY = ""
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Resolve Environment') {
            steps {
                script {
                    // Multibranch sets BRANCH_NAME; plain pipeline sets GIT_BRANCH
                    // (e.g. origin/develop) and checks out a detached HEAD.
                    def branch = env.BRANCH_NAME ?: env.GIT_BRANCH ?: sh(
                        script: 'git rev-parse --abbrev-ref HEAD',
                        returnStdout: true
                    ).trim()

                    branch = branch.replaceFirst(/^(refs\/heads\/|refs\/remotes\/)?origin\//, '')

                    env.GIT_BRANCH_RESOLVED = branch

                    switch (branch) {
                        case 'main':
                            env.COMPOSE_FILE_PATH = 'docker-compose.prod.yml'
                            env.COMPOSE_PROJECT   = 'leaf-api'
                            env.VAULT_ENV         = 'prod'
                            env.TAG               = 'latest'
                            env.API_HEALTH_URL    = 'http://127.0.0.1:16870/health'
                            env.SHOULD_DEPLOY     = 'true'
                            break
                        case 'develop':
                            env.COMPOSE_FILE_PATH = 'docker-compose.dev.yml'
                            env.COMPOSE_PROJECT   = 'leaf-api-dev'
                            env.VAULT_ENV         = 'develop'
                            env.TAG               = 'develop'
                            env.API_HEALTH_URL    = 'http://127.0.0.1:15870/health'
                            env.SHOULD_DEPLOY     = 'true'
                            break
                        default:
                            env.COMPOSE_FILE_PATH = 'docker-compose.dev.yml'
                            env.COMPOSE_PROJECT   = "leaf-api-ci-${branch.toLowerCase().replaceAll(/[^a-z0-9]/, '-')}"
                            env.VAULT_ENV         = 'develop'
                            env.TAG               = 'ci'
                            env.SHOULD_DEPLOY     = 'false'
                    }

                    // Every compose call combines the shared file with the
                    // environment override, so keep them together in one var.
                    env.COMPOSE_FILES = "-f docker-compose.yml -f ${env.COMPOSE_FILE_PATH}"

                    echo """
                    ===== Leaf-Classification Pipeline =====
                    Branch       : ${env.GIT_BRANCH_RESOLVED}
                    Compose file : ${env.COMPOSE_FILE_PATH}
                    Project      : ${env.COMPOSE_PROJECT}
                    Vault path   : ${env.VAULT_MOUNT}/${env.VAULT_ENV}/${env.VAULT_SECRET}
                    Image tag    : ${env.TAG}
                    Deploy       : ${env.SHOULD_DEPLOY}
                    ========================================
                    """.stripIndent()
                }
            }
        }

        stage('Fetch Secrets from Vault') {
            steps {
                withCredentials([string(credentialsId: 'VAULT_TOKEN', variable: 'VAULT_TOKEN')]) {
                    sh '''
                        set -e
                        umask 077   # the env file must stay readable only by jenkins

                        if [ ! -r "$VAULT_CACERT" ]; then
                            echo "ERROR: cannot read CA cert at $VAULT_CACERT"
                            exit 1
                        fi

                        RESP="$WORKSPACE/vault_${VAULT_SECRET}.json"

                        curl -fsS --cacert "$VAULT_CACERT" --resolve "$VAULT_RESOLVE" \
                            -H "X-Vault-Token: $VAULT_TOKEN" \
                            "$VAULT_ADDR/v1/$VAULT_MOUNT/data/$VAULT_ENV/$VAULT_SECRET" > "$RESP"

                        # Accept both KV v1 and KV v2 payload shapes.
                        jq -e '((.data.data // .data // {}) | type) == "object"' "$RESP" >/dev/null || {
                            echo "ERROR: unexpected Vault payload at $VAULT_MOUNT/$VAULT_ENV/$VAULT_SECRET"
                            sed -E 's/(X-Vault-Token:|token=|password=)[^[:space:]]+/\\1[REDACTED]/g' "$RESP"
                            exit 1
                        }

                        jq -r '((.data.data // .data // {}) | to_entries[] | (.key + "=" + (.value|tostring)))' \
                            "$RESP" > "$WORKSPACE/.env"

                        [ -s "$WORKSPACE/.env" ] || { echo "ERROR: .env is empty after parsing Vault payload"; exit 1; }

                        echo "--- .env keys (values hidden) ---"
                        grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' "$WORKSPACE/.env"

                        rm -f "$RESP"

                        # Fail here rather than three stages later with a confusing
                        # symptom: the app refuses to start without a database, and
                        # entrypoint.sh pulls the model weights from S3 before boot.
                        for key in DB_HOST DB_USER DB_PASSWORD DB_NAME JWT_SECRET \
                                   S3_BUCKET AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_DEFAULT_REGION; do
                            grep -q "^${key}=." "$WORKSPACE/.env" || { echo "ERROR: missing $key in Vault ($VAULT_ENV/$VAULT_SECRET)"; exit 1; }
                        done

                        # compose interpolates these when tagging the built image.
                        echo "TAG=$TAG"           >> "$WORKSPACE/.env"
                        echo "REGISTRY=$REGISTRY" >> "$WORKSPACE/.env"
                    '''
                }
            }
        }

        stage('Build') {
            steps {
                sh 'docker compose -p "$COMPOSE_PROJECT" $COMPOSE_FILES build'
            }
        }

        stage('Test') {
            // Built from the Dockerfile's `test` target, which adds tests/ on
            // top of the same base layer the runtime image uses. The suite
            // mocks the database and the models, so it needs no services.
            steps {
                sh '''
                    set -e
                    docker build \
                        --target test \
                        -t "leaf-api-test:${BUILD_NUMBER}" \
                        .

                    # Capped: the suite mocks the models, so it needs almost
                    # nothing, and this build shares a 2-core host with the
                    # running prod container. (The image BUILD cannot be capped
                    # this way — BuildKit rejects --cpus and ignores --memory;
                    # constrain the builder on the host instead, see README.)
                    docker run --rm --memory=1g --cpus=1 \
                        "leaf-api-test:${BUILD_NUMBER}" pytest -q tests
                '''
            }
            post {
                always {
                    sh 'docker image rm -f "leaf-api-test:${BUILD_NUMBER}" || true'
                }
            }
        }

        stage('Push') {
            when { expression { env.REGISTRY?.trim() } }
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'REGISTRY_CREDENTIALS',
                    usernameVariable: 'REG_USER',
                    passwordVariable: 'REG_PASS'
                )]) {
                    sh '''
                        echo "$REG_PASS" | docker login "$(echo "$REGISTRY" | cut -d/ -f1)" -u "$REG_USER" --password-stdin
                        docker compose -p "$COMPOSE_PROJECT" $COMPOSE_FILES push
                        docker logout "$(echo "$REGISTRY" | cut -d/ -f1)"
                    '''
                }
            }
        }

        stage('Deploy') {
            when { expression { env.SHOULD_DEPLOY == 'true' } }
            steps {
                sh '''
                    set -e
                    if [ -n "$REGISTRY" ]; then
                        docker compose -p "$COMPOSE_PROJECT" $COMPOSE_FILES pull
                    fi
                    docker compose -p "$COMPOSE_PROJECT" $COMPOSE_FILES up -d --remove-orphans

                    # The previous release's images are now untagged and unused.
                    # Dangling-only (no -a): images belonging to the other stacks
                    # on this host are still tagged and must survive.
                    docker image prune -f

                    # Cap the BuildKit cache instead of letting every build add
                    # another layer to it.
                    docker builder prune -f --keep-storage 2GB || true
                '''
            }
        }

        stage('Health Check') {
            when { expression { env.SHOULD_DEPLOY == 'true' } }
            steps {
                script {
                    // A host with no cached artifacts converts the classifiers to
                    // TFLite and exports YOLO to OpenVINO before serving, which is
                    // why compose allows a 900s start_period. Match that here.
                    retry(50) {
                        sleep(20)
                        def api = sh(
                            script: 'curl -sf "$API_HEALTH_URL" -o /dev/null -w \'%{http_code}\'',
                            returnStdout: true
                        ).trim()
                        if (api != '200') {
                            error("API health check failed — HTTP ${api}")
                        }
                        echo "Health check passed — API ${api}"
                    }
                }
            }
        }

        stage('Verify') {
            when { expression { env.SHOULD_DEPLOY == 'true' } }
            steps {
                sh '''
                    echo "===== Container Status ====="
                    docker compose -p "$COMPOSE_PROJECT" $COMPOSE_FILES \
                        ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
                '''
            }
        }
    }

    post {
        success {
            echo "Pipeline succeeded on branch: ${env.GIT_BRANCH_RESOLVED}"
        }
        failure {
            echo "Pipeline failed on branch: ${env.GIT_BRANCH_RESOLVED}"
        }
        always {
            // Secrets are written into the workspace; never leave them behind.
            sh 'rm -f "$WORKSPACE/.env" "$WORKSPACE"/vault_*.json || true'
            cleanWs()
        }
    }
}
