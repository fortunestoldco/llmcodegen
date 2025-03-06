#!/bin/bash

# Define checkpoint files
CHECKPOINT_DIR=".storybook_checkpoints"
TABLES_CHECKPOINT="$CHECKPOINT_DIR/tables_created"
IAM_CHECKPOINT="$CHECKPOINT_DIR/iam_created"
LAMBDA_CHECKPOINT="$CHECKPOINT_DIR/lambda_created"
API_CHECKPOINT="$CHECKPOINT_DIR/api_created"
KB_CHECKPOINT="$CHECKPOINT_DIR/kb_created"
ACTIONS_CHECKPOINT="$CHECKPOINT_DIR/actions_created"
AGENTS_CHECKPOINT="$CHECKPOINT_DIR/agents_created"

# Function to create checkpoint directory
setup_checkpoints() {
    mkdir -p "$CHECKPOINT_DIR"
}

# Function to check if a checkpoint exists
checkpoint_exists() {
    [ -f "$1" ]
}

# Function to create a checkpoint
create_checkpoint() {
    touch "$1"
}

set -e

# Default configuration
DEFAULT_PROJECT_NAME="storybook"
DEFAULT_REGION="us-east-1"  # Changed to us-east-1 as it has better Bedrock support
CONFIG_FILE=".storybook_config"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Load configuration if exists
load_config() {
    if [ -f "$CONFIG_FILE" ]; then
        source "$CONFIG_FILE"
    else
        PROJECT_NAME=$DEFAULT_PROJECT_NAME
        REGION=$DEFAULT_REGION
        CLAUDE_MODEL="anthropic.claude-3-sonnet-20240229-v1:0"

        # Save default config
        save_config
    fi
}

# Save configuration
save_config() {
    cat > "$CONFIG_FILE" << EOF
# Storybook System Configuration
PROJECT_NAME="$PROJECT_NAME"
REGION="$REGION"
CLAUDE_MODEL="$CLAUDE_MODEL"
EOF
}

# Update configuration
update_config() {
    clear
    echo -e "${BLUE}=== Configuration Settings ===${NC}"
    echo -e "Current settings:"
    echo -e "1. Project Name: ${GREEN}$PROJECT_NAME${NC}"
    echo -e "2. AWS Region: ${GREEN}$REGION${NC}"
    echo -e "3. Foundation Model: ${GREEN}$CLAUDE_MODEL${NC}"
    echo -e "4. Return to main menu"
    echo

    read -p "Select a setting to change (1-4): " config_choice

    case $config_choice in
        1)
            read -p "Enter new project name: " new_project_name
            if [ ! -z "$new_project_name" ]; then
                PROJECT_NAME=$new_project_name
                save_config
                echo -e "${GREEN}Project name updated.${NC}"
            fi
            ;;
        2)
            echo "Available regions with Bedrock support:"
            echo "1. us-east-1 (N. Virginia)"
            echo "2. us-west-2 (Oregon)"
            echo "3. eu-central-1 (Frankfurt)"
            echo "4. ap-northeast-1 (Tokyo)"
            read -p "Select a region (1-4): " region_choice
            case $region_choice in
                1) REGION="us-east-1" ;;
                2) REGION="us-west-2" ;;
                3) REGION="eu-central-1" ;;
                4) REGION="ap-northeast-1" ;;
                *) echo -e "${RED}Invalid choice. Region not changed.${NC}" ;;
            esac
            save_config
            echo -e "${GREEN}Region updated.${NC}"
            ;;
        3)
            echo "Available Foundation Models:"
            echo "1. Claude 3 Sonnet (anthropic.claude-3-sonnet-20240229-v1:0)"
            echo "2. Claude 3 Haiku (anthropic.claude-3-haiku-20240307-v1:0)"
            echo "3. Claude 3 Opus (anthropic.claude-3-opus-20240229-v1:0)"
            read -p "Select a model (1-3): " model_choice
            case $model_choice in
                1) CLAUDE_MODEL="anthropic.claude-3-sonnet-20240229-v1:0" ;;
                2) CLAUDE_MODEL="anthropic.claude-3-haiku-20240307-v1:0" ;;
                3) CLAUDE_MODEL="anthropic.claude-3-opus-20240229-v1:0" ;;
                *) echo -e "${RED}Invalid choice. Model not changed.${NC}" ;;
            esac
            save_config
            echo -e "${GREEN}Foundation model updated.${NC}"
            ;;
        4)
            return
            ;;
        *)
            echo -e "${RED}Invalid choice.${NC}"
            ;;
    esac

    read -p "Press Enter to continue..."
    update_config
}

# Check prerequisites
check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}Error: AWS CLI is not installed. Please install it first.${NC}"
        return 1
    fi

    # Check jq
    if ! command -v jq &> /dev/null; then
        echo -e "${RED}Error: jq is not installed. Please install it first.${NC}"
        return 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        echo -e "${RED}Error: AWS credentials not configured. Please run 'aws configure' first.${NC}"
        return 1
    fi

    # Check for necessary AWS Bedrock model access
    echo -e "${BLUE}Checking model access...${NC}"
    MODEL_ACCESS=$(aws bedrock list-foundation-models --region $REGION)
    CLAUDE_ACCESS=$(echo $MODEL_ACCESS | grep -c "$CLAUDE_MODEL" || true)

    if [ $CLAUDE_ACCESS -eq 0 ]; then
        echo -e "${RED}Warning: Cannot find $CLAUDE_MODEL access in region $REGION. You may need to request access in the AWS console.${NC}"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 1
        fi
    fi

    # Get ACCOUNT_ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)

    return 0
}

# Deploy the storybook system
deploy_system() {
    setup_checkpoints
    
    if ! check_prerequisites; then
        echo -e "${RED}Prerequisites check failed. Cannot deploy.${NC}"
        return 1
    fi

    # Get account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)

    echo -e "${BLUE}=== Deploying storybook system using AWS Bedrock ===${NC}"
    echo -e "${BLUE}Project: $PROJECT_NAME${NC}"
    echo -e "${BLUE}Region: $REGION${NC}"
    echo -e "${BLUE}Model: $CLAUDE_MODEL${NC}"

    # Create S3 bucket for assets
    BUCKET_NAME="${PROJECT_NAME}-assets-${ACCOUNT_ID}"
    
    if ! checkpoint_exists "$TABLES_CHECKPOINT"; then
        echo -e "${BLUE}Creating S3 bucket $BUCKET_NAME for assets...${NC}"
        aws s3api create-bucket \
            --bucket $BUCKET_NAME \
            --region $REGION \
            $(if [ "$REGION" != "us-east-1" ]; then echo "--create-bucket-configuration LocationConstraint=$REGION"; fi) \
            || echo -e "${RED}Bucket already exists or cannot be created. Continuing...${NC}"

        # Enable bucket versioning
        aws s3api put-bucket-versioning \
            --bucket $BUCKET_NAME \
            --versioning-configuration Status=Enabled

        # Create DynamoDB tables for storing state
        echo -e "${BLUE}Creating DynamoDB tables for state management...${NC}"

        # Project state table
        aws dynamodb create-table \
            --table-name storybook-projects \
            --attribute-definitions AttributeName=project_id,AttributeType=S \
            --key-schema AttributeName=project_id,KeyType=HASH \
            --billing-mode PAY_PER_REQUEST \
            --region $REGION \
            || echo -e "${RED}Projects table already exists. Continuing...${NC}"

        # Chapter table
        aws dynamodb create-table \
            --table-name storybook-chapters \
            --attribute-definitions \
                AttributeName=project_id,AttributeType=S \
                AttributeName=chapter_id,AttributeType=S \
            --key-schema \
                AttributeName=project_id,KeyType=HASH \
                AttributeName=chapter_id,KeyType=RANGE \
            --billing-mode PAY_PER_REQUEST \
            --region $REGION \
            || echo -e "${RED}Chapters table already exists. Continuing...${NC}"

        # Character table
        aws dynamodb create-table \
            --table-name storybook-characters \
            --attribute-definitions \
                AttributeName=project_id,AttributeType=S \
                AttributeName=character_id,AttributeType=S \
            --key-schema \
                AttributeName=project_id,KeyType=HASH \
                AttributeName=character_id,KeyType=RANGE \
            --billing-mode PAY_PER_REQUEST \
            --region $REGION \
            || echo -e "${RED}Characters table already exists. Continuing...${NC}"
            
        create_checkpoint "$TABLES_CHECKPOINT"
    else
        echo -e "${GREEN}Tables already created. Continuing...${NC}"
    fi

    if ! checkpoint_exists "$IAM_CHECKPOINT"; then
        # Create a Lambda execution role
        echo -e "${BLUE}Creating Lambda execution role...${NC}"
        ROLE_NAME="${PROJECT_NAME}-lambdarole"

        # Create a policy document for the trust relationship
        cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

        # Create the IAM role
        aws iam create-role \
            --role-name $ROLE_NAME \
            --assume-role-policy-document file://trust-policy.json \
            || echo -e "${RED}Role already exists. Continuing...${NC}"

        # Attach policies to the role
        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole \
            || echo -e "${RED}Failed to attach Lambda execution policy. Continuing...${NC}"

        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess \
            || echo -e "${RED}Failed to attach DynamoDB policy. Continuing...${NC}"

        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess \
            || echo -e "${RED}Failed to attach S3 policy. Continuing...${NC}"

        # Create a custom policy for Bedrock access
        cat > bedrock-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:*",
                "bedrock-agent:*",
                "bedrock-runtime:*",
                "bedrock-agent-runtime:*"
            ],
            "Resource": "*"
        }
    ]
}
EOF

        aws iam create-policy \
            --policy-name ${PROJECT_NAME}-bedrockpolicy \
            --policy-document file://bedrock-policy.json \
            || echo -e "${RED}Bedrock policy already exists. Continuing...${NC}"

        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-bedrockpolicy \
            || echo -e "${RED}Failed to attach Bedrock policy. Continuing...${NC}"

        # Create a policy document for the Knowledge Base service role trust relationship
        cat > kb-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "bedrock.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

        # Create the Knowledge Base service role
        KB_ROLE_NAME="${PROJECT_NAME}-kbrole"
        aws iam create-role \
            --role-name $KB_ROLE_NAME \
            --assume-role-policy-document file://kb-trust-policy.json \
            || echo -e "${RED}Knowledge Base role already exists. Continuing...${NC}"

        # Attach policies for Knowledge Base access to S3
        aws iam attach-role-policy \
            --role-name $KB_ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess \
            || echo -e "${RED}Failed to attach S3 read policy to Knowledge Base role. Continuing...${NC}"

        # Wait for IAM role to propagate
        echo -e "${BLUE}Waiting for IAM role to propagate...${NC}"
        sleep 30  # Increased from 10 to 30 seconds to ensure proper propagation
        
        create_checkpoint "$IAM_CHECKPOINT"
    else
        echo -e "${GREEN}IAM roles already created. Continuing...${NC}"
        ROLE_NAME="${PROJECT_NAME}-lambdarole"
        KB_ROLE_NAME="${PROJECT_NAME}-kbrole"
    fi

    if ! checkpoint_exists "$LAMBDA_CHECKPOINT"; then
        # Create Lambda files for each phase's orchestrator
        # 1. Create orchestrator Lambda code
        echo -e "${BLUE}Creating orchestrator Lambda code...${NC}"
        mkdir -p lambda
        cat > lambda/orchestrator.py << EOF
import json
import boto3
import os
import uuid
from datetime import datetime

# Initialize AWS clients
bedrock_runtime = boto3.client('bedrock-runtime', region_name=os.environ.get('REGION', 'us-east-1'))
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=os.environ.get('REGION', 'us-east-1'))
dynamodb = boto3.resource('dynamodb', region_name=os.environ.get('REGION', 'us-east-1'))

# Define tables
projects_table = dynamodb.Table('storybook-projects')
chapters_table = dynamodb.Table('storybook-chapters')
characters_table = dynamodb.Table('storybook-characters')

# Define agent IDs (to be filled with the actual IDs after creation)
AGENT_IDS = {
    "executive_director": os.environ.get('EXECUTIVE_DIRECTOR_AGENT_ID', ''),
    "creative_director": os.environ.get('CREATIVE_DIRECTOR_AGENT_ID', ''),
    "human_feedback_manager": os.environ.get('HUMAN_FEEDBACK_MANAGER_AGENT_ID', ''),
    "quality_assessment_director": os.environ.get('QUALITY_ASSESSMENT_AGENT_ID', ''),
    "project_timeline_manager": os.environ.get('TIMELINE_MANAGER_AGENT_ID', ''),
    "market_alignment_director": os.environ.get('MARKET_ALIGNMENT_AGENT_ID', ''),
    "structure_architect": os.environ.get('STRUCTURE_ARCHITECT_AGENT_ID', ''),
    "plot_development_specialist": os.environ.get('PLOT_DEVELOPMENT_SPECIALIST_AGENT_ID', ''),
    "world_building_expert": os.environ.get('WORLD_BUILDING_EXPERT_AGENT_ID', ''),
    "character_psychology_specialist": os.environ.get('CHARACTER_PSYCHOLOGY_SPECIALIST_AGENT_ID', ''),
    "character_voice_designer": os.environ.get('CHARACTER_VOICE_DESIGNER_AGENT_ID', ''),
    "character_relationship_mapper": os.environ.get('CHARACTER_RELATIONSHIP_MAPPER_AGENT_ID', ''),
    "domain_knowledge_specialist": os.environ.get('DOMAIN_KNOWLEDGE_SPECIALIST_AGENT_ID', ''),
    "cultural_authenticity_expert": os.environ.get('CULTURAL_AUTHENTICITY_EXPERT_AGENT_ID', ''),
    "content_development_director": os.environ.get('CONTENT_DEVELOPMENT_DIRECTOR_AGENT_ID', ''),
    "chapter_drafters": os.environ.get('CHAPTER_DRAFTERS_AGENT_ID', ''),
    "scene_construction_specialists": os.environ.get('SCENE_CONSTRUCTION_SPECIALISTS_AGENT_ID', ''),
    "dialogue_crafters": os.environ.get('DIALOGUE_CRAFTERS_AGENT_ID', ''),
    "continuity_manager": os.environ.get('CONTINUITY_MANAGER_AGENT_ID', ''),
    "voice_consistency_monitor": os.environ.get('VOICE_CONSISTENCY_MONITOR_AGENT_ID', ''),
    "emotional_arc_designer": os.environ.get('EMOTIONAL_ARC_DESIGNER_AGENT_ID', ''),
    "editorial_director": os.environ.get('EDITORIAL_DIRECTOR_AGENT_ID', ''),
    "structural_editor": os.environ.get('STRUCTURAL_EDITOR_AGENT_ID', ''),
    "character_arc_evaluator": os.environ.get('CHARACTER_ARC_EVALUATOR_AGENT_ID', ''),
    "thematic_coherence_analyst": os.environ.get('THEMATIC_COHERENCE_ANALYST_AGENT_ID', ''),
    "prose_enhancement_specialist": os.environ.get('PROSE_ENHANCEMENT_SPECIALIST_AGENT_ID', ''),
    "dialogue_refinement_expert": os.environ.get('DIALOGUE_REFINEMENT_EXPERT_AGENT_ID', ''),
    "rhythm_cadence_optimizer": os.environ.get('RHYTHM_CADENCE_OPTIMIZER_AGENT_ID', ''),
    "grammar_consistency_checker": os.environ.get('GRAMMAR_CONSISTENCY_CHECKER_AGENT_ID', ''),
    "fact_verification_specialist": os.environ.get('FACT_VERIFICATION_SPECIALIST_AGENT_ID', ''),
    "positioning_specialist": os.environ.get('POSITIONING_SPECIALIST_AGENT_ID', ''),
    "title_blurb_optimizer": os.environ.get('TITLE_BLURB_OPTIMIZER_AGENT_ID', ''),
    "differentiation_strategist": os.environ.get('DIFFERENTIATION_STRATEGIST_AGENT_ID', ''),
    "formatting_standards_expert": os.environ.get('FORMATTING_STANDARDS_EXPERT_AGENT_ID', '')
}

# Define agent alias IDs
AGENT_ALIAS_IDS = {
    "executive_director": os.environ.get('EXECUTIVE_DIRECTOR_ALIAS_ID', ''),
    "creative_director": os.environ.get('CREATIVE_DIRECTOR_ALIAS_ID', ''),
    "human_feedback_manager": os.environ.get('HUMAN_FEEDBACK_MANAGER_ALIAS_ID', ''),
    "quality_assessment_director": os.environ.get('QUALITY_ASSESSMENT_ALIAS_ID', ''),
    "content_development_director": os.environ.get('CONTENT_DEVELOPMENT_DIRECTOR_ALIAS_ID', ''),
    "structure_architect": os.environ.get('STRUCTURE_ARCHITECT_ALIAS_ID', '')
}

def lambda_handler(event, context):
    """Main handler for orchestrating the storybook system."""

    # Parse input
    body = json.loads(event.get('body', '{}'))

    # Determine operation to perform
    operation = body.get('operation', 'default')

    if operation == 'create_project':
        return create_new_project(body)
    elif operation == 'invoke_agent':
        return invoke_agent(body)
    elif operation == 'transition_phase':
        return transition_phase(body)
    elif operation == 'store_chapter':
        return store_chapter(body)
    elif operation == 'store_character':
        return store_character(body)
    elif operation == 'retrieve_project':
        return retrieve_project(body)
    else:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid operation'})
        }

def create_new_project(body):
    """Create a new novel project."""

    project_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # Extract project data
    title = body.get('title', 'Untitled Novel')
    synopsis = body.get('synopsis', '')
    genre = body.get('genre', ['fiction'])
    target_audience = body.get('target_audience', ['adult'])

    # Create project record
    project = {
        'project_id': project_id,
        'title': title,
        'synopsis': synopsis,
        'genre': genre,
        'target_audience': target_audience,
        'phase': 'initialization',
        'status': 'active',
        'created_at': timestamp,
        'updated_at': timestamp,
        'quality_assessment': {}
    }

    # Store in DynamoDB
    projects_table.put_item(Item=project)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'project_id': project_id,
            'message': 'Project created successfully',
            'project': project
        })
    }

def invoke_agent(body):
    """Invoke a specific agent to process a task."""

    project_id = body.get('project_id')
    agent_name = body.get('agent')
    task = body.get('task')

    if not project_id or not agent_name or not task:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Missing required parameters'})
        }

    # Get project data
    project_response = projects_table.get_item(Key={'project_id': project_id})
    if 'Item' not in project_response:
        return {
            'statusCode': 404,
            'body': json.dumps({'error': 'Project not found'})
        }

    project = project_response['Item']

    # Check if agent exists
    agent_id = AGENT_IDS.get(agent_name)
    alias_id = AGENT_ALIAS_IDS.get(agent_name)

    if not agent_id:
        return {
            'statusCode': 404,
            'body': json.dumps({'error': 'Agent not found'})
        }

    if not alias_id:
        alias_id = "Production" # Use a default if specific alias ID not found

    # Prepare input for the agent
    input_text = json.dumps({
        'project_id': project_id,
        'project': project,
        'task': task,
        'phase': project['phase']
    })

    # Invoke Bedrock agent
    try:
        response = bedrock_agent_runtime.invoke_agent(
            agentId=agent_id,
            agentAliasId=alias_id,
            sessionId=project_id,
            inputText=f"Task: {task}\\nProject Context: {project['title']} - {project['synopsis']}"
        )

        # Process agent response
        completion = response.get('completion', {})
        response_text = completion.get('text', 'No response from agent')

        # Update project with agent's output
        timestamp = datetime.now().isoformat()
        if 'agent_outputs' not in project:
            project['agent_outputs'] = {}

        if agent_name not in project['agent_outputs']:
            project['agent_outputs'][agent_name] = []

        project['agent_outputs'][agent_name].append({
            'timestamp': timestamp,
            'task': task,
            'response': response_text
        })

        project['updated_at'] = timestamp
        project['current_agent'] = agent_name

        # Save updated project
        projects_table.put_item(Item=project)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'project_id': project_id,
                'agent': agent_name,
                'response': response_text
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

def transition_phase(body):
    """Transition a project to a new phase."""

    project_id = body.get('project_id')
    new_phase = body.get('new_phase')

    valid_phases = ['initialization', 'development', 'creation', 'refinement', 'finalization', 'complete']

    if not project_id or new_phase not in valid_phases:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid project_id or phase'})
        }

    # Get project data
    project_response = projects_table.get_item(Key={'project_id': project_id})
    if 'Item' not in project_response:
        return {
            'statusCode': 404,
            'body': json.dumps({'error': 'Project not found'})
        }

    project = project_response['Item']
    current_phase = project['phase']
    timestamp = datetime.now().isoformat()

    # Check if transition is valid
    phase_order = {phase: idx for idx, phase in enumerate(valid_phases)}
    if phase_order.get(new_phase, 99) <= phase_order.get(current_phase, 0):
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid phase transition. Cannot move backward.'})
        }

    # Update phase history
    if 'phase_history' not in project:
        project['phase_history'] = {}

    if current_phase not in project['phase_history']:
        project['phase_history'][current_phase] = []

    project['phase_history'][current_phase].append({
        'end_time': timestamp,
        'transition_to': new_phase,
        'quality_assessment': project.get('quality_assessment', {})
    })

    # Update project phase
    project['phase'] = new_phase
    project['updated_at'] = timestamp

    # Save updated project
    projects_table.put_item(Item=project)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'project_id': project_id,
            'previous_phase': current_phase,
            'current_phase': new_phase,
            'message': f'Project transitioned from {current_phase} to {new_phase}'
        })
    }

def store_chapter(body):
    """Store a chapter in the database."""
    project_id = body.get('project_id')
    chapter_id = body.get('chapter_id', str(uuid.uuid4()))
    title = body.get('title')
    content = body.get('content')

    if not project_id or not title or not content:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Missing required parameters'})
        }

    # Create chapter record
    timestamp = datetime.now().isoformat()
    chapter = {
        'project_id': project_id,
        'chapter_id': chapter_id,
        'title': title,
        'content': content,
        'created_at': timestamp,
        'updated_at': timestamp
    }

    # Store in DynamoDB
    chapters_table.put_item(Item=chapter)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'project_id': project_id,
            'chapter_id': chapter_id,
            'message': 'Chapter stored successfully'
        })
    }

def store_character(body):
    """Store a character in the database."""
    project_id = body.get('project_id')
    character_id = body.get('character_id', str(uuid.uuid4()))
    name = body.get('name')
    description = body.get('description', '')
    role = body.get('role', '')

    if not project_id or not name:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Missing required parameters'})
        }

    # Create character record
    timestamp = datetime.now().isoformat()
    character = {
        'project_id': project_id,
        'character_id': character_id,
        'name': name,
        'description': description,
        'role': role,
        'created_at': timestamp,
        'updated_at': timestamp
    }

    # Store in DynamoDB
    characters_table.put_item(Item=character)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'project_id': project_id,
            'character_id': character_id,
            'message': 'Character stored successfully'
        })
    }

def retrieve_project(body):
    """Retrieve a project from the database."""
    project_id = body.get('project_id')

    if not project_id:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Missing project_id parameter'})
        }

    # Get project data
    project_response = projects_table.get_item(Key={'project_id': project_id})
    if 'Item' not in project_response:
        return {
            'statusCode': 404,
            'body': json.dumps({'error': 'Project not found'})
        }

    project = project_response['Item']

    # Get chapters
    chapter_response = chapters_table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key('project_id').eq(project_id)
    )
    chapters = chapter_response.get('Items', [])

    # Get characters
    character_response = characters_table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key('project_id').eq(project_id)
    )
    characters = character_response.get('Items', [])

    # Combine data
    project_data = {
        'project': project,
        'chapters': chapters,
        'characters': characters
    }

    return {
        'statusCode': 200,
        'body': json.dumps(project_data)
    }
EOF

        # Create Lambda function
        echo -e "${BLUE}Creating Lambda functions...${NC}"

        # Zip the code for Lambda
        cd lambda
        zip -r orchestrator.zip orchestrator.py
        cd ..

        # Get the IAM role ARN
        ROLE_ARN=$(aws iam get-role --role-name ${ROLE_NAME} --query 'Role.Arn' --output text)

        # Get the Knowledge Base role ARN
        KB_ROLE_ARN=$(aws iam get-role --role-name ${KB_ROLE_NAME} --query 'Role.Arn' --output text)

        # Verify that we have a valid role ARN before proceeding
        if [ -z "$ROLE_ARN" ]; then
            echo -e "${RED}Error: Could not retrieve role ARN. Please check if the role was created correctly.${NC}"
            return 1
        fi

        echo -e "${BLUE}Using IAM role: $ROLE_ARN${NC}"
        echo -e "${BLUE}Using Knowledge Base role: $KB_ROLE_ARN${NC}"

        # Create the Lambda function
        LAMBDA_ARN=$(aws lambda create-function \
            --function-name ${PROJECT_NAME}-orchestrator \
            --runtime python3.10 \
            --role $ROLE_ARN \
            --handler orchestrator.lambda_handler \
            --zip-file fileb://lambda/orchestrator.zip \
            --timeout 300 \
            --environment "Variables={REGION=$REGION}" \
            --region $REGION \
            --query 'FunctionArn' --output text \
            || aws lambda update-function-code \
               --function-name ${PROJECT_NAME}-orchestrator \
               --zip-file fileb://lambda/orchestrator.zip \
               --region $REGION \
               --query 'FunctionArn' --output text)

        echo -e "${GREEN}Lambda function created: $LAMBDA_ARN${NC}"
        
        create_checkpoint "$LAMBDA_CHECKPOINT"
    else
        echo -e "${GREEN}Lambda function already created. Continuing...${NC}"
        # Get the Lambda ARN
        LAMBDA_ARN=$(aws lambda get-function --function-name ${PROJECT_NAME}-orchestrator --region $REGION --query 'Configuration.FunctionArn' --output text)
        
        # Get the IAM role ARN
        ROLE_ARN=$(aws iam get-role --role-name ${ROLE_NAME} --query 'Role.Arn' --output text)

        # Get the Knowledge Base role ARN
        KB_ROLE_ARN=$(aws iam get-role --role-name ${KB_ROLE_NAME} --query 'Role.Arn' --output text)
    fi

    if ! checkpoint_exists "$API_CHECKPOINT"; then
        # Create API Gateway
        echo -e "${BLUE}Creating API Gateway...${NC}"
        API_ID=$(aws apigateway create-rest-api \
            --name ${PROJECT_NAME}-api \
            --description "API for storybook System" \
            --region $REGION \
            --query 'id' --output text \
            || echo "Failed to create API Gateway. It may already exist.")

        if [ ! -z "$API_ID" ]; then
            # Get the root resource ID
            ROOT_RESOURCE_ID=$(aws apigateway get-resources \
                --rest-api-id $API_ID \
                --region $REGION \
                --query 'items[0].id' --output text)

            # Create a resource
            RESOURCE_ID=$(aws apigateway create-resource \
                --rest-api-id $API_ID \
                --parent-id $ROOT_RESOURCE_ID \
                --path-part "storybook" \
                --region $REGION \
                --query 'id' --output text \
                || echo "Resource may already exist")

            # Create POST method
            aws apigateway put-method \
                --rest-api-id $API_ID \
                --resource-id $RESOURCE_ID \
                --http-method POST \
                --authorization-type NONE \
                --region $REGION \
                || echo "Method may already exist"

            # Set up the integration with Lambda
            aws apigateway put-integration \
                --rest-api-id $API_ID \
                --resource-id $RESOURCE_ID \
                --http-method POST \
                --type AWS_PROXY \
                --integration-http-method POST \
                --uri arn:aws:apigateway:${REGION}:lambda:path/2015-03-31/functions/${LAMBDA_ARN}/invocations \
                --region $REGION \
                || echo "Integration may already exist"

            # Deploy the API
            DEPLOYMENT_ID=$(aws apigateway create-deployment \
                --rest-api-id $API_ID \
                --stage-name prod \
                --region $REGION \
                --query 'id' --output text)

            # Add permission for API Gateway to invoke Lambda
            aws lambda add-permission \
                --function-name ${PROJECT_NAME}-orchestrator \
                --statement-id apigateway-prod-$(date +%s) \
                --action lambda:InvokeFunction \
                --principal apigateway.amazonaws.com \
                --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_ID}/*/POST/storybook" \
                --region $REGION \
                || echo "Permission may already exist"

            echo -e "${GREEN}API Gateway deployed. Endpoint: https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod/storybook${NC}"
        else
            echo -e "${RED}Failed to create or retrieve API Gateway. Continuing...${NC}"
        fi
        
        create_checkpoint "$API_CHECKPOINT"
    else
        echo -e "${GREEN}API Gateway already created. Continuing...${NC}"
        # Get the API Gateway ID
        API_ID=$(aws apigateway get-rest-apis --region $REGION --query "items[?name=='${PROJECT_NAME}-api'].id" --output text)
    fi

    if ! checkpoint_exists "$KB_CHECKPOINT"; then
        # Create Bedrock Knowledge Base for project reference materials
        echo -e "${BLUE}Creating Amazon Bedrock Knowledge Base...${NC}"

        # Create a directory for knowledge base files
        mkdir -p kb_files
        cat > kb_files/novel_writing_guide.txt << EOF
# Novel Writing Guide

## Structure
- Three-act structure: setup, confrontation, resolution
- Five-act structure: exposition, rising action, climax, falling action, resolution
- Hero's journey: ordinary world, call to adventure, refusal, meeting the mentor, crossing the threshold, tests/allies/enemies, approach, ordeal, reward, road back, resurrection, return

## Character Development
- Give characters clear motivations
- Create internal and external conflicts
- Develop character arcs showing growth or change
- Use character backstory to inform decisions

## Plot Development
- Create a compelling inciting incident
- Build rising tension throughout
- Plant setups and payoffs
- Create meaningful stakes
- Design a satisfying climax and resolution

## Genre Conventions
- Romance: Meet cute, obstacles, dark moment, resolution
- Mystery: Crime, investigation, red herrings, resolution
- Fantasy: World-building, magic systems, epic stakes
- Thriller: Ticking clock, high stakes, twists
EOF

        # Upload knowledge base files to S3
        aws s3 cp kb_files/novel_writing_guide.txt s3://$BUCKET_NAME/kb_files/novel_writing_guide.txt

        # Check if knowledge base creation is supported in the region
        KB_SUPPORTED=$(aws bedrock-agent help create-knowledge-base 2>&1 | grep -c "not available" || true)

        if [ $KB_SUPPORTED -eq 0 ]; then
            echo -e "${BLUE}Creating Knowledge Base...${NC}"

            # Create an OpenSearch collection first (fixes the auto issue)
            # Check if OpenSearch Serverless is available in the region
            OS_SUPPORTED=$(aws opensearchserverless help create-collection 2>&1 | grep -c "not available" || true)
            
            if [ $OS_SUPPORTED -eq 0 ]; then
                # Create a security policy for encryption
                aws opensearchserverless create-security-policy \
                    --name ${PROJECT_NAME}-encryption-policy \
                    --type encryption \
                    --policy "{\"Rules\":[{\"ResourceType\":\"collection\",\"Resource\":[\"collection/${PROJECT_NAME}-vectors\"]}],\"AWSOwnedKey\":true}" \
                    --region $REGION \
                    || echo "Encryption policy may already exist"
                
                # Create a security policy for network
                aws opensearchserverless create-security-policy \
                    --name ${PROJECT_NAME}-network-policy \
                    --type network \
                    --policy "{\"Rules\":[{\"ResourceType\":\"collection\",\"Resource\":[\"collection/${PROJECT_NAME}-vectors\"],\"AllowFromPublic\":true}]}" \
                    --region $REGION \
                    || echo "Network policy may already exist"
                
                # Create the OpenSearch collection
                COLLECTION_ID=$(aws opensearchserverless create-collection \
                    --name ${PROJECT_NAME}-vectors \
                    --type VECTORSEARCH \
                    --region $REGION \
                    --query 'createCollectionDetail.id' --output text \
                    || echo "Collection may already exist")
                
                # Wait for collection to be created
                echo -e "${BLUE}Waiting for OpenSearch collection to be created...${NC}"
                aws opensearchserverless batch-get-collection \
                    --names ${PROJECT_NAME}-vectors \
                    --region $REGION \
                    --query 'collectionDetails[0].status' \
                    --output text \
                    || echo "Failed to get collection status"
                
                # Keep checking until it's ACTIVE or timeout after 10 minutes
                timeout=600
                start_time=$(date +%s)
                while true; do
                    current_time=$(date +%s)
                    elapsed=$((current_time - start_time))
                    
                    if [ $elapsed -gt $timeout ]; then
                        echo -e "${RED}Timed out waiting for collection to become active.${NC}"
                        break
                    fi
                    
                    status=$(aws opensearchserverless batch-get-collection \
                        --names ${PROJECT_NAME}-vectors \
                        --region $REGION \
                        --query 'collectionDetails[0].status' \
                        --output text)
                    
                    if [ "$status" == "ACTIVE" ]; then
                        echo -e "${GREEN}OpenSearch collection is now active.${NC}"
                        break
                    elif [ "$status" == "FAILED" ]; then
                        echo -e "${RED}OpenSearch collection creation failed.${NC}"
                        break
                    else
                        echo -e "${BLUE}Collection status: $status. Waiting...${NC}"
                        sleep 30
                    fi
                done
                
                # Get the collection ARN
                COLLECTION_ARN=$(aws opensearchserverless batch-get-collection \
                    --names ${PROJECT_NAME}-vectors \
                    --region $REGION \
                    --query 'collectionDetails[0].arn' \
                    --output text)
                
                if [ ! -z "$COLLECTION_ARN" ]; then
                    # Create a vector knowledge base with OpenSearch Serverless configuration
                    KB_ID=$(aws bedrock-agent create-knowledge-base \
                        --name "${PROJECT_NAME}-knowledgebase" \
                        --description "Knowledge base for novel writing" \
                        --role-arn "$KB_ROLE_ARN" \
                        --knowledge-base-configuration "{\"type\":\"VECTOR\",\"vectorKnowledgeBaseConfiguration\":{\"embeddingModelArn\":\"arn:aws:bedrock:${REGION}::foundation-model/amazon.titan-embed-text-v1\"}}" \
                        --storage-configuration "{\"type\":\"OPENSEARCH_SERVERLESS\",\"opensearchServerlessConfiguration\":{\"collectionArn\":\"$COLLECTION_ARN\",\"vectorIndexName\":\"storybook-vectors\",\"fieldMapping\":{\"metadataField\":\"metadata\",\"textField\":\"text\",\"vectorField\":\"vector\"}}}" \
                        --region $REGION \
                        --query 'knowledgeBase.knowledgeBaseId' --output text \
                        || echo "Failed to create Knowledge Base")
                else
                    echo -e "${RED}Failed to get OpenSearch collection ARN. Continuing without Knowledge Base...${NC}"
                    KB_ID=""
                fi
            else
                echo -e "${RED}OpenSearch Serverless is not available in this region. Using default RDS.${NC}"
                # Use RDS storage configuration instead
                KB_ID=$(aws bedrock-agent create-knowledge-base \
                    --name "${PROJECT_NAME}-knowledgebase" \
                    --description "Knowledge base for novel writing" \
                    --role-arn "$KB_ROLE_ARN" \
                    --knowledge-base-configuration "{\"type\":\"VECTOR\",\"vectorKnowledgeBaseConfiguration\":{\"embeddingModelArn\":\"arn:aws:bedrock:${REGION}::foundation-model/amazon.titan-embed-text-v1\"}}" \
                    --storage-configuration "{\"type\":\"RDS\",\"rdsConfiguration\":{\"resourceArn\":\"auto\"}}" \
                    --region $REGION \
                    --query 'knowledgeBase.knowledgeBaseId' --output text \
                    || echo "Failed to create Knowledge Base")
            fi

            if [[ -z "$KB_ID" || "$KB_ID" == "Failed to create Knowledge Base" ]]; then
                echo -e "${RED}Failed to create Knowledge Base. Continuing without it...${NC}"
                KB_ID=""
            else
                echo -e "${GREEN}Knowledge Base created: $KB_ID${NC}"

                # Create a data source for the knowledge base
                DS_ID=$(aws bedrock-agent create-data-source \
                    --knowledge-base-id $KB_ID \
                    --name "${PROJECT_NAME}-datasource" \
                    --description "Novel writing guides and reference materials" \
                    --data-source-configuration "{\"type\":\"S3\",\"s3Configuration\":{\"bucketName\":\"$BUCKET_NAME\",\"inclusionPrefixes\":[\"kb_files/\"]}}" \
                    --region $REGION \
                    --query 'dataSource.dataSourceId' --output text)

                if [ ! -z "$DS_ID" ]; then
                    echo -e "${GREEN}Data Source created: $DS_ID${NC}"

                    # Start data ingestion
                    aws bedrock-agent start-ingestion-job \
                        --knowledge-base-id $KB_ID \
                        --data-source-id $DS_ID \
                        --region $REGION \
                        || echo -e "${RED}Failed to start ingestion job. Continuing...${NC}"

                    echo -e "${GREEN}Ingestion job started for Knowledge Base${NC}"
                else
                    echo -e "${RED}Failed to create Data Source. Continuing...${NC}"
                fi
            fi
        else
            echo -e "${RED}Knowledge Base creation is not supported in this region. Skipping...${NC}"
            KB_ID="" # Set KB_ID to empty to avoid errors later
        fi
        
        create_checkpoint "$KB_CHECKPOINT"
    else
        echo -e "${GREEN}Knowledge Base already created. Continuing...${NC}"
        
        # Try to get KB_ID if it exists
        KB_SUPPORTED=$(aws bedrock-agent help create-knowledge-base 2>&1 | grep -c "not available" || true)
        if [ $KB_SUPPORTED -eq 0 ]; then
            KB_LIST=$(aws bedrock-agent list-knowledge-bases --region $REGION)
            KB_ID=$(echo $KB_LIST | jq -r ".knowledgeBaseSummaries[] | select(.name==\"${PROJECT_NAME}-knowledgebase\") | .knowledgeBaseId" 2>/dev/null || echo "")
        else
            KB_ID=""
        fi
    fi

    if ! checkpoint_exists "$ACTIONS_CHECKPOINT"; then
        # Create the OpenAPI schema files for action groups
        mkdir -p schemas

        # Create schema for DynamoDB operations
        cat > schemas/dynamo_schema.json << EOF
{
  "openapi": "3.0.0",
  "info": {
    "title": "DynamoDB API",
    "version": "1.0.0"
  },
  "paths": {
    "/storeChapter": {
      "post": {
        "operationId": "storeChapter",
        "description": "Store a chapter in the database",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "project_id": {
                    "type": "string",
                    "description": "The ID of the project"
                  },
                  "chapter_id": {
                    "type": "string",
                    "description": "The ID of the chapter"
                  },
                  "title": {
                    "type": "string",
                    "description": "The title of the chapter"
                  },
                  "content": {
                    "type": "string",
                    "description": "The content of the chapter"
                  }
                },
                "required": ["project_id", "chapter_id", "title", "content"]
              }
            }
          }
        }
      }
    },
    "/storeCharacter": {
      "post": {
        "operationId": "storeCharacter",
        "description": "Store a character in the database",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "project_id": {
                    "type": "string",
                    "description": "The ID of the project"
                  },
                  "character_id": {
                    "type": "string",
                    "description": "The ID of the character"
                  },
                  "name": {
                    "type": "string",
                    "description": "The name of the character"
                  },
                  "description": {
                    "type": "string",
                    "description": "The description of the character"
                  },
                  "role": {
                    "type": "string",
                    "description": "The role of the character (protagonist, antagonist, etc.)"
                  }
                },
                "required": ["project_id", "character_id", "name"]
              }
            }
          }
        }
      }
    },
    "/retrieveProject": {
      "post": {
        "operationId": "retrieveProject",
        "description": "Retrieve a project from the database",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "project_id": {
                    "type": "string",
                    "description": "The ID of the project to retrieve"
                  }
                },
                "required": ["project_id"]
              }
            }
          }
        }
      }
    }
  }
}
EOF

        # Create schema for Research Tools
        cat > schemas/research_schema.json << EOF
{
  "openapi": "3.0.0",
  "info": {
    "title": "Research API",
    "version": "1.0.0"
  },
  "paths": {
    "/domainResearch": {
      "post": {
        "operationId": "domainResearch",
        "description": "Conduct research on a specific domain",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "topic": {
                    "type": "string",
                    "description": "The topic to research"
                  },
                  "depth": {
                    "type": "string",
                    "description": "The depth of research (basic, detailed, comprehensive)"
                  }
                },
                "required": ["topic"]
              }
            }
          }
        }
      }
    },
    "/marketResearch": {
      "post": {
        "operationId": "marketResearch",
        "description": "Conduct market research for a novel",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "genre": {
                    "type": "string",
                    "description": "The genre of the novel"
                  },
                  "target_audience": {
                    "type": "string",
                    "description": "The target audience for the novel"
                  }
                },
                "required": ["genre"]
              }
            }
          }
        }
      }
    },
    "/factCheck": {
      "post": {
        "operationId": "factCheck",
        "description": "Verify factual information",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "claim": {
                    "type": "string",
                    "description": "The claim to verify"
                  },
                  "context": {
                    "type": "string",
                    "description": "The context of the claim"
                  }
                },
                "required": ["claim"]
              }
            }
          }
        }
      }
    }
  }
}
EOF

        # Create schema for Quality Assessment
        cat > schemas/quality_schema.json << EOF
{
  "openapi": "3.0.0",
  "info": {
    "title": "Quality Assessment API",
    "version": "1.0.0"
  },
  "paths": {
    "/assessQuality": {
      "post": {
        "operationId": "assessQuality",
        "description": "Assess the quality of a novel element",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "project_id": {
                    "type": "string",
                    "description": "The ID of the project"
                  },
                  "element_type": {
                    "type": "string",
                    "description": "The type of element (chapter, character, plot, etc.)"
                  },
                  "element_id": {
                    "type": "string",
                    "description": "The ID of the element"
                  },
                  "content": {
                    "type": "string",
                    "description": "The content to assess"
                  }
                },
                "required": ["project_id", "element_type", "content"]
              }
            }
          }
        }
      }
    },
    "/checkQualityGate": {
      "post": {
        "operationId": "checkQualityGate",
        "description": "Check if a quality gate is passed",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "project_id": {
                    "type": "string",
                    "description": "The ID of the project"
                  },
                  "gate_name": {
                    "type": "string",
                    "description": "The name of the quality gate"
                  }
                },
                "required": ["project_id", "gate_name"]
              }
            }
          }
        }
      }
    }
  }
}
EOF

        # Create schema for Timeline Management
        cat > schemas/timeline_schema.json << EOF
{
  "openapi": "3.0.0",
  "info": {
    "title": "Timeline Management API",
    "version": "1.0.0"
  },
  "paths": {
    "/createMilestone": {
      "post": {
        "operationId": "createMilestone",
        "description": "Create a milestone in the project timeline",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "project_id": {
                    "type": "string",
                    "description": "The ID of the project"
                  },
                  "milestone_name": {
                    "type": "string",
                    "description": "The name of the milestone"
                  },
                  "description": {
                    "type": "string",
                    "description": "The description of the milestone"
                  },
                  "due_date": {
                    "type": "string",
                    "description": "The due date for the milestone (ISO format)"
                  }
                },
                "required": ["project_id", "milestone_name"]
              }
            }
          }
        }
      }
    },
    "/updateMilestoneStatus": {
      "post": {
        "operationId": "updateMilestoneStatus",
        "description": "Update the status of a milestone",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "project_id": {
                    "type": "string",
                    "description": "The ID of the project"
                  },
                  "milestone_name": {
                    "type": "string",
                    "description": "The name of the milestone"
                  },
                  "status": {
                    "type": "string",
                    "description": "The status of the milestone (not_started, in_progress, completed, delayed)"
                  }
                },
                "required": ["project_id", "milestone_name", "status"]
              }
            }
          }
        }
      }
    }
  }
}
EOF

        # Upload schemas to S3
        aws s3 cp schemas/dynamo_schema.json s3://$BUCKET_NAME/schemas/dynamo_schema.json
        aws s3 cp schemas/research_schema.json s3://$BUCKET_NAME/schemas/research_schema.json
        aws s3 cp schemas/quality_schema.json s3://$BUCKET_NAME/schemas/quality_schema.json
        aws s3 cp schemas/timeline_schema.json s3://$BUCKET_NAME/schemas/timeline_schema.json

        # Check if Bedrock Agent API is supported in this region
        AG_SUPPORTED=$(aws bedrock-agent help create-agent-action-group 2>&1 | grep -c "not available" || true)

        if [ $AG_SUPPORTED -ne 0 ]; then
            echo -e "${RED}Bedrock Agent API is not available in this region ($REGION). Skipping agent creation...${NC}"
            echo -e "${RED}You may need to switch to a region where Bedrock Agent is available, such as us-east-1 or us-west-2${NC}"

            # Clean up temporary files
            echo -e "${BLUE}Cleaning up temporary files...${NC}"
            rm -f trust-policy.json bedrock-policy.json kb-trust-policy.json
            rm -rf lambda kb_files schemas

            echo -e "${YELLOW}=== storybook System partially deployed! ===${NC}"
            echo -e "${YELLOW}Lambda function and DynamoDB tables created but Bedrock agents were not created.${NC}"
            if [ ! -z "$API_ID" ]; then
                echo -e "${GREEN}API Endpoint: https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod/storybook${NC}"
            fi
            return 0
        fi

        # Create common action groups that all agents will use
        echo -e "${BLUE}Creating common action groups...${NC}"

        # Create a Bedrock agent for testing
        echo -e "${BLUE}Creating a test agent to get agentId and agentVersion...${NC}"
        TEST_AGENT_CONFIG='{
            "agentName": "TestAgent-'$(date +%s)'",
            "agentResourceRoleArn": "'$ROLE_ARN'",
            "foundationModel": "'$CLAUDE_MODEL'",
            "description": "Test agent for getting proper agent ID and version",
            "instruction": "You are a test agent. Just acknowledge any request."
        }'

        echo $TEST_AGENT_CONFIG > test_agent_config.json
        
        # Create test agent
        TEST_AGENT_ID=$(aws bedrock-agent create-agent \
            --cli-input-json file://test_agent_config.json \
            --region $REGION \
            --query 'agent.agentId' --output text \
            || echo "Failed to create test agent")

        if [[ "$TEST_AGENT_ID" == "Failed to create test agent" || -z "$TEST_AGENT_ID" ]]; then
            echo -e "${RED}Failed to create test agent. Cannot create action groups. Check your permissions.${NC}"
            rm -f test_agent_config.json
            return 1
        fi

        # Create test agent version
        TEST_AGENT_VERSION=$(aws bedrock-agent create-agent-version \
            --agent-id $TEST_AGENT_ID \
            --region $REGION \
            --query 'agentVersion' --output text \
            || echo "Failed to create test agent version")

        if [[ "$TEST_AGENT_VERSION" == "Failed to create test agent version" || -z "$TEST_AGENT_VERSION" ]]; then
            echo -e "${RED}Failed to create test agent version. Cannot create action groups.${NC}"
            # Clean up test agent
            aws bedrock-agent delete-agent --agent-id $TEST_AGENT_ID --skip-resource-in-use-check --region $REGION || echo "Failed to delete test agent"
            rm -f test_agent_config.json
            return 1
        fi

        echo -e "${GREEN}Successfully created test agent ($TEST_AGENT_ID) with version $TEST_AGENT_VERSION${NC}"

        # Create an action group for DynamoDB operations - CORRECTED FORMAT
        cat > dynamo_action_group.json << EOF
{
  "actionGroupName": "DynamoDBOperations",
  "description": "Actions for interacting with DynamoDB to store and retrieve novel data",
  "actionGroupExecutor": {
    "lambda": "${LAMBDA_ARN}"
  },
  "apiSchema": {
    "s3": {
      "s3BucketName": "${BUCKET_NAME}",
      "s3ObjectKey": "schemas/dynamo_schema.json"
    }
  },
  "agentId": "${TEST_AGENT_ID}",
  "agentVersion": "${TEST_AGENT_VERSION}"
}
EOF

        # Create action group for the knowledge base
        if [ ! -z "$KB_ID" ]; then
            cat > kb_action_group.json << EOF
{
  "actionGroupName": "WritingKnowledgeBase",
  "description": "Action group for accessing novel writing knowledge base",
  "actionGroupExecutor": {
    "knowledgeBaseId": "${KB_ID}"
  },
  "agentId": "${TEST_AGENT_ID}",
  "agentVersion": "${TEST_AGENT_VERSION}"
}
EOF
        fi

        # Create action group for research tools - CORRECTED FORMAT
        cat > research_action_group.json << EOF
{
  "actionGroupName": "ResearchTools",
  "description": "Actions for conducting research",
  "actionGroupExecutor": {
    "lambda": "${LAMBDA_ARN}"
  },
  "apiSchema": {
    "s3": {
      "s3BucketName": "${BUCKET_NAME}",
      "s3ObjectKey": "schemas/research_schema.json"
    }
  },
  "agentId": "${TEST_AGENT_ID}",
  "agentVersion": "${TEST_AGENT_VERSION}"
}
EOF

        # Create action group for quality assessment - CORRECTED FORMAT
        cat > quality_action_group.json << EOF
{
  "actionGroupName": "QualityAssessment",
  "description": "Actions for assessing novel quality",
  "actionGroupExecutor": {
    "lambda": "${LAMBDA_ARN}"
  },
  "apiSchema": {
    "s3": {
      "s3BucketName": "${BUCKET_NAME}",
      "s3ObjectKey": "schemas/quality_schema.json"
    }
  },
  "agentId": "${TEST_AGENT_ID}",
  "agentVersion": "${TEST_AGENT_VERSION}"
}
EOF

        # Create action group for timeline management - CORRECTED FORMAT
        cat > timeline_action_group.json << EOF
{
  "actionGroupName": "TimelineManagement",
  "description": "Actions for managing project timeline",
  "actionGroupExecutor": {
    "lambda": "${LAMBDA_ARN}"
  },
  "apiSchema": {
    "s3": {
      "s3BucketName": "${BUCKET_NAME}",
      "s3ObjectKey": "schemas/timeline_schema.json"
    }
  },
  "agentId": "${TEST_AGENT_ID}",
  "agentVersion": "${TEST_AGENT_VERSION}"
}
EOF

        # Create action groups
        echo -e "${BLUE}Creating action groups...${NC}"

        DYNAMO_AG_ID=$(aws bedrock-agent create-agent-action-group \
            --cli-input-json file://dynamo_action_group.json \
            --region $REGION \
            --query 'agentActionGroup.agentActionGroupId' --output text \
            || echo "Failed to create DynamoDB action group")

        if [[ ! -z "$DYNAMO_AG_ID" && "$DYNAMO_AG_ID" != "Failed to create DynamoDB action group" ]]; then
            echo -e "${GREEN}Created DynamoDB action group with ID: $DYNAMO_AG_ID${NC}"
        else
            echo -e "${RED}Failed to create DynamoDB action group. Continuing...${NC}"
            DYNAMO_AG_ID=""
        fi

        KB_AG_ID=""
        if [ ! -z "$KB_ID" ] && [ "$KB_ID" != "Failed to create Knowledge Base" ]; then
            KB_AG_ID=$(aws bedrock-agent create-agent-action-group \
                --cli-input-json file://kb_action_group.json \
                --region $REGION \
                --query 'agentActionGroup.agentActionGroupId' --output text \
                || echo "Failed to create Knowledge Base action group")

            if [[ ! -z "$KB_AG_ID" && "$KB_AG_ID" != "Failed to create Knowledge Base action group" ]]; then
                echo -e "${GREEN}Created Knowledge Base action group with ID: $KB_AG_ID${NC}"
            else
                echo -e "${RED}Failed to create Knowledge Base action group. Continuing...${NC}"
                KB_AG_ID=""
            fi
        else
            echo -e "${RED}Skipping creation of Knowledge Base action group due to earlier failure${NC}"
        fi

        RESEARCH_AG_ID=$(aws bedrock-agent create-agent-action-group \
            --cli-input-json file://research_action_group.json \
            --region $REGION \
            --query 'agentActionGroup.agentActionGroupId' --output text \
            || echo "Failed to create Research Tools action group")

        if [[ ! -z "$RESEARCH_AG_ID" && "$RESEARCH_AG_ID" != "Failed to create Research Tools action group" ]]; then
            echo -e "${GREEN}Created Research Tools action group with ID: $RESEARCH_AG_ID${NC}"
        else
            echo -e "${RED}Failed to create Research Tools action group. Continuing...${NC}"
            RESEARCH_AG_ID=""
        fi

        QUALITY_AG_ID=$(aws bedrock-agent create-agent-action-group \
            --cli-input-json file://quality_action_group.json \
            --region $REGION \
            --query 'agentActionGroup.agentActionGroupId' --output text \
            || echo "Failed to create Quality Assessment action group")

        if [[ ! -z "$QUALITY_AG_ID" && "$QUALITY_AG_ID" != "Failed to create Quality Assessment action group" ]]; then
            echo -e "${GREEN}Created Quality Assessment action group with ID: $QUALITY_AG_ID${NC}"
        else
            echo -e "${RED}Failed to create Quality Assessment action group. Continuing...${NC}"
            QUALITY_AG_ID=""
        fi

        TIMELINE_AG_ID=$(aws bedrock-agent create-agent-action-group \
            --cli-input-json file://timeline_action_group.json \
            --region $REGION \
            --query 'agentActionGroup.agentActionGroupId' --output text \
            || echo "Failed to create Timeline Management action group")

        if [[ ! -z "$TIMELINE_AG_ID" && "$TIMELINE_AG_ID" != "Failed to create Timeline Management action group" ]]; then
            echo -e "${GREEN}Created Timeline Management action group with ID: $TIMELINE_AG_ID${NC}"
        else
            echo -e "${RED}Failed to create Timeline Management action group. Continuing...${NC}"
            TIMELINE_AG_ID=""
        fi

        # Clean up test agent after creating action groups
        aws bedrock-agent delete-agent --agent-id $TEST_AGENT_ID --skip-resource-in-use-check --region $REGION || echo "Failed to delete test agent"
        rm -f test_agent_config.json
        
        create_checkpoint "$ACTIONS_CHECKPOINT"
    else
        echo -e "${GREEN}Action groups already created. Continuing...${NC}"
        
        # These values will be populated properly in the actual script
        DYNAMO_AG_ID="existing-dynamo-ag-id"
        KB_AG_ID=""
        if [ ! -z "$KB_ID" ]; then
            KB_AG_ID="existing-kb-ag-id"
        fi
        RESEARCH_AG_ID="existing-research-ag-id"
        QUALITY_AG_ID="existing-quality-ag-id"
        TIMELINE_AG_ID="existing-timeline-ag-id"
    fi

    if ! checkpoint_exists "$AGENTS_CHECKPOINT"; then
        # Function to create a Bedrock agent
        create_bedrock_agent() {
            local agent_name=$1
            local description=$2
            local instruction=$3
            local action_groups=$4

            echo -e "${BLUE}Creating $agent_name agent...${NC}"

            # Create agent with proper JSON escaping for instruction
            INSTRUCTION_JSON=$(echo "$instruction" | jq -R -s .)

            # Create agent configuration JSON file
            cat > agent_config.json << EOF
{
  "agentName": "$agent_name",
  "agentResourceRoleArn": "${ROLE_ARN}",
  "foundationModel": "${CLAUDE_MODEL}",
  "description": "$description",
  "instruction": $INSTRUCTION_JSON
}
EOF

            # Create agent
            local agent_id=$(aws bedrock-agent create-agent \
                --cli-input-json file://agent_config.json \
                --region $REGION \
                --query 'agent.agentId' --output text \
                || echo "Failed to create agent")

            if [[ -z "$agent_id" || "$agent_id" == "Failed to create agent" ]]; then
                echo -e "${RED}Failed to create $agent_name agent. Skipping.${NC}"
                echo "FAILED FAILED"
                return
            fi

            echo -e "${GREEN}Created $agent_name agent with ID: $agent_id${NC}"

            local alias_id=""

            # Create a draft version first
            local version=$(aws bedrock-agent create-agent-version \
                --agent-id $agent_id \
                --region $REGION \
                --query 'agentVersion' --output text \
                || echo "Failed to create agent version")

            if [[ ! -z "$version" && "$version" != "Failed to create agent version" ]]; then
                echo -e "${GREEN}Created agent version: $version${NC}"

                # Associate action groups with the agent version
                for action_group in $action_groups; do
                    if [[ ! -z "$action_group" && "$action_group" != "Failed to create"* ]]; then
                        echo -e "${BLUE}Associating action group $action_group with $agent_name...${NC}"
                        aws bedrock-agent associate-agent-action-group \
                            --agent-id $agent_id \
                            --agent-version "$version" \
                            --action-group-id $action_group \
                            --region $REGION \
                            || echo -e "${RED}Failed to associate action group $action_group with $agent_name${NC}"

                        # Add a short delay to ensure association is complete
                        sleep 2
                    fi
                done

                # Create alias
                local alias_name="Production"
                alias_id=$(aws bedrock-agent create-agent-alias \
                    --agent-id $agent_id \
                    --agent-alias-name $alias_name \
                    --routing-configuration "[{\"agentVersion\":\"$version\"}]" \
                    --region $REGION \
                    --query 'agentAlias.agentAliasId' --output text \
                    || echo "Failed to create agent alias")

                if [[ ! -z "$alias_id" && "$alias_id" != "Failed to create agent alias" ]]; then
                    echo -e "${GREEN}Created alias '$alias_name' with ID: $alias_id${NC}"
                else
                    echo -e "${RED}Failed to create alias for $agent_name.${NC}"
                    alias_id="FAILED"
                fi
            else
                echo -e "${RED}Failed to create version for $agent_name.${NC}"
                version="FAILED"
                alias_id="FAILED"
            fi

            # Return both agent ID and alias ID separated by a space
            echo "$agent_id $alias_id"
        }

        # Create all agents using the function
        # Start with fewer agents first to test functionality

        # Add KB_AG_ID conditionally to the action groups
        KB_AG_PARAM=""
        if [ ! -z "$KB_AG_ID" ]; then
            KB_AG_PARAM="$KB_AG_ID"
        fi

        # Check if we have at least one action group
        if [[ -z "$DYNAMO_AG_ID" && -z "$KB_AG_ID" && -z "$RESEARCH_AG_ID" && -z "$QUALITY_AG_ID" && -z "$TIMELINE_AG_ID" ]]; then
            echo -e "${RED}No action groups were created successfully. Cannot create agents.${NC}"
            echo -e "${RED}Please check your AWS Bedrock permissions and try again.${NC}"

            # Clean up temporary files
            echo -e "${BLUE}Cleaning up temporary files...${NC}"
            rm -f trust-policy.json bedrock-policy.json kb-trust-policy.json dynamo_action_group.json kb_action_group.json research_action_group.json quality_action_group.json timeline_action_group.json agent_config.json
            rm -rf lambda kb_files schemas

            echo -e "${YELLOW}=== storybook System partially deployed! ===${NC}"
            echo -e "${YELLOW}Lambda function and DynamoDB tables created but Bedrock agents were not created.${NC}"
            if [ ! -z "$API_ID" ]; then
                echo -e "${GREEN}API Endpoint: https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod/storybook${NC}"
            fi
            return 0
        fi

        # Phase 1: Initialization phase agents
        echo -e "${BLUE}Creating initialization phase agents...${NC}"
        EXEC_DIRECTOR_RESULT=$(create_bedrock_agent \
            "ExecutiveDirector" \
            "Oversees the entire novel creation process and delegates tasks." \
            "You are the Executive Director, responsible for overseeing the entire novel creation process. You coordinate all team members and ensure the project stays on track. Your primary responsibilities include: delegating tasks, making high-level decisions, resolving conflicts, monitoring progress, and serving as the final decision-maker. Always maintain a strategic view of the project." \
            "$DYNAMO_AG_ID $TIMELINE_AG_ID $QUALITY_AG_ID")

        read EXEC_DIRECTOR_ID EXEC_DIRECTOR_ALIAS_ID <<< "$EXEC_DIRECTOR_RESULT"

        HUMAN_FEEDBACK_RESULT=$(create_bedrock_agent \
            "HumanFeedbackManager" \
            "Processes and integrates feedback from human reviewers." \
            "You are the Human Feedback Manager, responsible for processing input from human reviewers and readers. Your responsibilities include interpreting feedback constructively, identifying patterns, prioritizing based on importance, suggesting implementable changes, and balancing different perspectives. Always provide concrete ways to incorporate feedback into the novel." \
            "$DYNAMO_AG_ID")

        read HUMAN_FEEDBACK_MANAGER_ID HUMAN_FEEDBACK_MANAGER_ALIAS_ID <<< "$HUMAN_FEEDBACK_RESULT"

        QUALITY_ASSESSMENT_RESULT=$(create_bedrock_agent \
            "QualityAssessmentDirector" \
            "Evaluates the quality of the novel at various stages." \
            "You are the Quality Assessment Director, responsible for evaluating the novel against quality standards. Your responsibilities include defining quality metrics, conducting comprehensive assessments, identifying areas for improvement, tracking quality trends, and determining if quality gates are met for phase transitions. Provide objective assessments with clear metrics." \
            "$DYNAMO_AG_ID $QUALITY_AG_ID")

        read QUALITY_ASSESSMENT_DIRECTOR_ID QUALITY_ASSESSMENT_ALIAS_ID <<< "$QUALITY_ASSESSMENT_RESULT"

        # Phase 2: Development phase agents - create just a few key ones for demonstration
        echo -e "${BLUE}Creating development phase agents...${NC}"
        CREATIVE_DIRECTOR_RESULT=$(create_bedrock_agent \
            "CreativeDirector" \
            "Manages creative aspects including story, characters, and setting." \
            "You are the Creative Director, responsible for the artistic vision of the novel. You manage all creative elements including story concept, character development, and setting design. Your goal is to ensure creative cohesion while encouraging innovation. Work closely with the Executive Director and specialized agents to maintain a unified creative vision." \
            "$DYNAMO_AG_ID $KB_AG_PARAM")

        read CREATIVE_DIRECTOR_ID CREATIVE_DIRECTOR_ALIAS_ID <<< "$CREATIVE_DIRECTOR_RESULT"

        STRUCTURE_ARCHITECT_RESULT=$(create_bedrock_agent \
            "StructureArchitect" \
            "Designs the novel's overall structure and pacing." \
            "You are the Structure Architect, responsible for designing the novel's overall structure. Your responsibilities include crafting the narrative structure, planning chapter organization and pacing, designing story arcs, ensuring structural integrity, and balancing exposition, conflict, and resolution throughout the novel." \
            "$DYNAMO_AG_ID $KB_AG_PARAM")

        read STRUCTURE_ARCHITECT_ID STRUCTURE_ARCHITECT_ALIAS_ID <<< "$STRUCTURE_ARCHITECT_RESULT"

        # Phase 3: Creation phase agents - create just one for demonstration
        echo -e "${BLUE}Creating creation phase agent...${NC}"
        CONTENT_DEV_RESULT=$(create_bedrock_agent \
            "ContentDevelopmentDirector" \
            "Oversees the development of content elements." \
            "You are the Content Development Director, responsible for coordinating the actual writing process. Your responsibilities include ensuring all content fits together coherently, maintaining consistency across different elements, translating outlines and plans into actual content, identifying content gaps, and managing the revision process for early drafts." \
            "$DYNAMO_AG_ID $KB_AG_PARAM")

        read CONTENT_DEVELOPMENT_DIRECTOR_ID CONTENT_DEVELOPMENT_ALIAS_ID <<< "$CONTENT_DEV_RESULT"

        # Update Lambda environment variables with all agent IDs
        echo -e "${BLUE}Updating Lambda environment variables with agent IDs...${NC}"

        # Filter out any failed agent creations
        if [[ "$EXEC_DIRECTOR_ID" != "FAILED" && ! -z "$EXEC_DIRECTOR_ID" ]]; then
            INIT_PHASE_AGENTS="EXECUTIVE_DIRECTOR_AGENT_ID=$EXEC_DIRECTOR_ID,EXECUTIVE_DIRECTOR_ALIAS_ID=$EXEC_DIRECTOR_ALIAS_ID"
        else
            INIT_PHASE_AGENTS=""
        fi

        if [[ "$HUMAN_FEEDBACK_MANAGER_ID" != "FAILED" && ! -z "$HUMAN_FEEDBACK_MANAGER_ID" ]]; then
            if [ ! -z "$INIT_PHASE_AGENTS" ]; then
                INIT_PHASE_AGENTS="$INIT_PHASE_AGENTS,HUMAN_FEEDBACK_MANAGER_AGENT_ID=$HUMAN_FEEDBACK_MANAGER_ID,HUMAN_FEEDBACK_MANAGER_ALIAS_ID=$HUMAN_FEEDBACK_MANAGER_ALIAS_ID"
            else
                INIT_PHASE_AGENTS="HUMAN_FEEDBACK_MANAGER_AGENT_ID=$HUMAN_FEEDBACK_MANAGER_ID,HUMAN_FEEDBACK_MANAGER_ALIAS_ID=$HUMAN_FEEDBACK_MANAGER_ALIAS_ID"
            fi
        fi

        if [[ "$QUALITY_ASSESSMENT_DIRECTOR_ID" != "FAILED" && ! -z "$QUALITY_ASSESSMENT_DIRECTOR_ID" ]]; then
            if [ ! -z "$INIT_PHASE_AGENTS" ]; then
                INIT_PHASE_AGENTS="$INIT_PHASE_AGENTS,QUALITY_ASSESSMENT_AGENT_ID=$QUALITY_ASSESSMENT_DIRECTOR_ID,QUALITY_ASSESSMENT_ALIAS_ID=$QUALITY_ASSESSMENT_ALIAS_ID"
            else
                INIT_PHASE_AGENTS="QUALITY_ASSESSMENT_AGENT_ID=$QUALITY_ASSESSMENT_DIRECTOR_ID,QUALITY_ASSESSMENT_ALIAS_ID=$QUALITY_ASSESSMENT_ALIAS_ID"
            fi
        fi

        # Development phase
        if [[ "$CREATIVE_DIRECTOR_ID" != "FAILED" && ! -z "$CREATIVE_DIRECTOR_ID" ]]; then
            DEV_PHASE_AGENTS="CREATIVE_DIRECTOR_AGENT_ID=$CREATIVE_DIRECTOR_ID,CREATIVE_DIRECTOR_ALIAS_ID=$CREATIVE_DIRECTOR_ALIAS_ID"
        else
            DEV_PHASE_AGENTS=""
        fi

        if [[ "$STRUCTURE_ARCHITECT_ID" != "FAILED" && ! -z "$STRUCTURE_ARCHITECT_ID" ]]; then
            if [ ! -z "$DEV_PHASE_AGENTS" ]; then
                DEV_PHASE_AGENTS="$DEV_PHASE_AGENTS,STRUCTURE_ARCHITECT_AGENT_ID=$STRUCTURE_ARCHITECT_ID,STRUCTURE_ARCHITECT_ALIAS_ID=$STRUCTURE_ARCHITECT_ALIAS_ID"
            else
                DEV_PHASE_AGENTS="STRUCTURE_ARCHITECT_AGENT_ID=$STRUCTURE_ARCHITECT_ID,STRUCTURE_ARCHITECT_ALIAS_ID=$STRUCTURE_ARCHITECT_ALIAS_ID"
            fi
        fi

        # Creation phase
        if [[ "$CONTENT_DEVELOPMENT_DIRECTOR_ID" != "FAILED" && ! -z "$CONTENT_DEVELOPMENT_DIRECTOR_ID" ]]; then
            CREATION_PHASE_AGENTS="CONTENT_DEVELOPMENT_DIRECTOR_AGENT_ID=$CONTENT_DEVELOPMENT_DIRECTOR_ID,CONTENT_DEVELOPMENT_DIRECTOR_ALIAS_ID=$CONTENT_DEVELOPMENT_ALIAS_ID"
        else
            CREATION_PHASE_AGENTS=""
        fi

        # Combine all agent IDs
        ALL_AGENT_IDS="REGION=$REGION"
        if [ ! -z "$INIT_PHASE_AGENTS" ]; then
            ALL_AGENT_IDS="$ALL_AGENT_IDS,$INIT_PHASE_AGENTS"
        fi
        if [ ! -z "$DEV_PHASE_AGENTS" ]; then
            ALL_AGENT_IDS="$ALL_AGENT_IDS,$DEV_PHASE_AGENTS"
        fi
        if [ ! -z "$CREATION_PHASE_AGENTS" ]; then
            ALL_AGENT_IDS="$ALL_AGENT_IDS,$CREATION_PHASE_AGENTS"
        fi

        # Update Lambda environment variables
        aws lambda update-function-configuration \
            --function-name ${PROJECT_NAME}-orchestrator \
            --environment "Variables={$ALL_AGENT_IDS}" \
            --region $REGION \
            || echo -e "${RED}Failed to update Lambda environment variables. Continuing...${NC}"
            
        create_checkpoint "$AGENTS_CHECKPOINT"
    else
        echo -e "${GREEN}Agents already created. Continuing...${NC}"
    fi

    # Clean up temporary files
    echo -e "${BLUE}Cleaning up temporary files...${NC}"
    rm -f trust-policy.json bedrock-policy.json kb-trust-policy.json dynamo_action_group.json kb_action_group.json research_action_group.json quality_action_group.json timeline_action_group.json agent_config.json
    rm -rf lambda kb_files schemas

    echo -e "${GREEN}=== storybook System deployed successfully! ===${NC}"
    echo -e "${GREEN}Use the API Gateway endpoint to interact with the system.${NC}"
    if [ ! -z "$API_ID" ]; then
        echo -e "${GREEN}API Endpoint: https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod/storybook${NC}"
    fi
    echo -e "${GREEN}Example curl command to create a new project:${NC}"
    echo -e "curl -X POST -H \"Content-Type: application/json\" -d '{\"operation\":\"create_project\",\"title\":\"My Novel\",\"synopsis\":\"A thrilling story about...\",\"genre\":[\"thriller\"],\"target_audience\":[\"young adult\"]}' https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod/storybook"
    echo -e "${GREEN}Example curl command to invoke an agent:${NC}"
    echo -e "curl -X POST -H \"Content-Type: application/json\" -d '{\"operation\":\"invoke_agent\",\"project_id\":\"YOUR_PROJECT_ID\",\"agent\":\"executive_director\",\"task\":\"Create an initial project plan for the novel\"}' https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod/storybook"

    return 0
}

# Update the prompts and configuration of existing agents
update_agents() {
    # Rest of the function remains the same
    # ...
}

# Delete the entire deployment
delete_deployment() {
    # Rest of the function remains the same
    # ...
}

# Main menu
show_menu() {
    clear
    echo -e "${BLUE}=============================================${NC}"
    echo -e "${BLUE}      Storybook AWS Bedrock Deployment      ${NC}"
    echo -e "${BLUE}=============================================${NC}"
    echo -e "Project: ${GREEN}$PROJECT_NAME${NC}"
    echo -e "Region: ${GREEN}$REGION${NC}"
    echo -e "Model: ${GREEN}$CLAUDE_MODEL${NC}"
    echo -e "${BLUE}=============================================${NC}"
    echo
    echo "1. Create New Deployment"
    echo "2. Update Existing Agents"
    echo "3. Delete Deployment"
    echo "4. Configuration Settings"
    echo "5. Resume Deployment"
    echo "6. Exit"
    echo
    read -p "Select an option (1-6): " choice

    case $choice in
        1)
            # Reset checkpoints to start fresh
            rm -rf "$CHECKPOINT_DIR"
            deploy_system
            read -p "Press Enter to continue..."
            ;;
        2)
            update_agents
            read -p "Press Enter to continue..."
            ;;
        3)
            delete_deployment
            read -p "Press Enter to continue..."
            ;;
        4)
            update_config
            ;;
        5)
            deploy_system
            read -p "Press Enter to continue..."
            ;;
        6)
            echo -e "${GREEN}Exiting...${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option. Please try again.${NC}"
            read -p "Press Enter to continue..."
            ;;
    esac
}

# Load configuration
load_config

# Create checkpoint directory
setup_checkpoints

# Main loop
while true; do
    show_menu
done
