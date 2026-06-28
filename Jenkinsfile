pipeline {
    agent any

    environment {
        VAULT_ADDR = 'http://127.0.0.1:8200'
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Fetch Secrets from Vault') {
            steps {
                withCredentials([string(credentialsId: 'vault-token', variable: 'VAULT_TOKEN')]) {
                    script {
                        def vaultPath = env.BRANCH_NAME == 'main' ? 'prod' : 'develop'
                        sh """
                            curl -sf \
                                -H "X-Vault-Token: \${VAULT_TOKEN}" \
                                ${VAULT_ADDR}/v1/yamwisdom/data/${vaultPath}/ml_api \
                            | jq -r '.data.data | to_entries[] | "\\(.key)=\\(.value)"' > .env
                        """
                    }
                }
            }
        }

        stage('Run Tests') {
            steps {
                sh '''
                    docker run --rm \
                        --env-file .env \
                        -v $(pwd):/app \
                        -w /app \
                        python:3.12-slim \
                        sh -c "pip install -q -r Requirement.txt && pytest tests/ -v --tb=short"
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    def tag = env.BRANCH_NAME == 'main' ? 'prod' : 'develop'
                    sh "docker build -t leaf-api:${tag} ."
                }
            }
        }

        stage('Deploy') {
            steps {
                script {
                    def override = env.BRANCH_NAME == 'main'
                        ? 'docker-compose.prod.yml'
                        : 'docker-compose.dev.yml'
                    sh """
                        docker compose -f docker-compose.yml -f ${override} up -d
                        rm -f .env
                    """
                }
            }
        }

    }

    post {
        always {
            sh 'rm -f .env'
            cleanWs()
        }
        success {
            echo "Deploy สำเร็จ — branch: ${env.BRANCH_NAME}"
        }
        failure {
            echo "Pipeline ล้มเหลว — branch: ${env.BRANCH_NAME} — ตรวจสอบ logs ด้านบน"
        }
    }
}
