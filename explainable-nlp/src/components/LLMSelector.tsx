import React, { useState } from 'react';
import { Button, Modal, Form, Accordion, Badge } from 'react-bootstrap';

// OpenAI models list
const openAIModels = [
  { name: "gpt-5-2025-08-07" },
  { name: "o4-mini-2025-04-16" },
  { name: "gpt-4.1-nano-2025-04-14" },
  { name: "gpt-3.5-turbo" },
  { name: "gpt-4o-mini-2024-07-18" },
  { name: "gpt-5-nano-2025-08-07" },
  { name: "gpt-5-mini-2025-08-07" }
];

// Ollama models list
const ollamaModels = [
  { name: "jsk/bio-mistral" },
  { name: "phi3.5:latest" },
  { name: "gemma:2b" },
  { name: "llama3.1:8b" },
  { name: "mistral:7b" }
];

// Groq models list
const groqModels = [
  { name: "allam-2-7b" },
  { name: "llama-3.3-70b-versatile" },
  { name: "llama-3.1-8b-instant" }
];

// OpenRouter models list
const openrouterModels = [
  { name: "deepseek/deepseek-r1-0528-qwen3-8b:free" },
  { name: "deepseek-r1-0528" },
  { name: "sarvam-m" },
  { name: "devstral-small" },
  { name: "gemma-3n-e4b-it" },
  { name: "google/gemma-3n-e2b-it:free" },
  { name: "deephermes-3-mistral-24b-preview" },
  { name: "phi-4-reasoning-plus" },
  { name: "phi-4-reasoning" },
  { name: "internvl3-14b" },
  { name: "internvl3-2b" },
  { name: "deepseek-prover-v2" },
  { name: "qwen3-30b-a3b" },
  { name: "qwen3-8b" },
  { name: "qwen3-14b" },
  { name: "qwen3-32b" },
  { name: "qwen3-235b-a22b" },
  { name: "deepseek-r1t-chimera" },
  { name: "mai-ds-r1" },
  { name: "glm-z1-32b" },
  { name: "glm-4-32b" },
  { name: "shisa-v2-llama3.3-70b" },
  { name: "qwq-32b-arliai-rpr-v1" },
  { name: "deepcoder-14b-preview" },
  { name: "kimi-vl-a3b-thinking" },
  { name: "llama-3.3-nemotron-super-49b-v1" },
  { name: "llama-3.1-nemotron-ultra-253b-v1" },
  { name: "llama-4-maverick" },
  { name: "llama-4-scout" },
  { name: "deepseek-v3-base" },
  { name: "qwen2.5-vl-3b-instruct" },
  { name: "gemini-2.5-pro-exp-03-25" },
  { name: "qwen2.5-vl-32b-instruct" },
  { name: "deepseek-chat-v3-0324" },
  { name: "qwerky-72b" },
  { name: "mistral-small-3.1-24b-instruct" },
  { name: "olympiccoder-32b" },
  { name: "gemma-3-1b-it" },
  { name: "gemma-3-4b-it" },
  { name: "gemma-3-12b-it" },
  { name: "reka-flash-3" },
  { name: "gemma-3-27b-it" },
  { name: "deepseek-r1-zero" },
  { name: "qwq-32b" },
  { name: "moonlight-16b-a3b-instruct" },
  { name: "deephermes-3-llama-3-8b-preview" },
  { name: "dolphin3.0-r1-mistral-24b" },
  { name: "dolphin3.0-mistral-24b" },
  { name: "qwen2.5-vl-72b-instruct" },
  { name: "mistral-small-24b-instruct-2501" },
  { name: "deepseek-r1-distill-qwen-32b" },
  { name: "deepseek-r1-distill-qwen-14b" },
  { name: "deepseek-r1-distill-llama-70b" },
  { name: "deepseek-r1" },
  { name: "deepseek-chat" },
  { name: "gemini-2.0-flash-exp" },
  { name: "llama-3.3-70b-instruct" },
  { name: "qwen-2.5-coder-32b-instruct" },
  { name: "qwen-2.5-7b-instruct" },
  { name: "llama-3.2-3b-instruct" },
  { name: "llama-3.2-1b-instruct" },
  { name: "llama-3.2-11b-vision-instruct" },
  { name: "qwen-2.5-72b-instruct" },
  { name: "qwen-2.5-vl-7b-instruct" },
  { name: "llama-3.1-405b" },
  { name: "llama-3.1-8b-instruct" },
  { name: "mistral-nemo" },
  { name: "gemma-2-9b-it" },
  { name: "mistral-7b-instruct" }
];

// Gemini models list
const geminiModels = [
  { name: "models/gemini-1.5-flash-8b" },
  { name: "gemini-1.5-flash" },
  { name: "gemini-2.0-flash-exp" },
  { name: "gemini-2.5-pro-exp-03-25" }
];

interface LLMSelectorProps {
  onModelsSubmit: (models: string[]) => Promise<void>;
  buttonText?: string;
  buttonVariant?: string;
  disabled?: boolean;
  currentModels?: Array<{ provider: string; model: string }>; // Current explanation models
}

interface LLMModel {
  provider: string;
  model: string;
}

const LLMSelector: React.FC<LLMSelectorProps> = ({
  onModelsSubmit,
  buttonText = "Choose Different LLMs",
  buttonVariant = "outline-primary",
  disabled = false,
  currentModels = []
}) => {
  const [showModal, setShowModal] = useState(false);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Convert current models to the format used by selectedModels (provider:model)
  // Note: model names in DB have underscores instead of dots, so we need to handle that
  const getCurrentModelKeys = (): string[] => {
    return currentModels.map(m => {
      // Convert model name back from underscore to dot format for matching
      const modelName = m.model.replace(/_/g, '.');
      return `${m.provider}:${modelName}`;
    });
  };

  const handleSubmit = async () => {
    if (selectedModels.length === 0) {
      alert('Please select at least one model.');
      return;
    }

    setIsSubmitting(true);
    try {
      await onModelsSubmit(selectedModels);
      setShowModal(false);
      setSelectedModels([]);
    } catch (error) {
      console.error('Failed to submit models:', error);
      alert('Failed to add explanation models. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleModalOpen = () => {
    // Initialize with current models when opening modal
    const currentKeys = getCurrentModelKeys();
    setSelectedModels(currentKeys);
    setShowModal(true);
  };

  const handleModalClose = () => {
    setShowModal(false);
    setSelectedModels([]);
  };

  const handleModelToggle = (modelKey: string, checked: boolean) => {
    const updated = checked
      ? [...selectedModels, modelKey]
      : selectedModels.filter((m) => m !== modelKey);
    setSelectedModels(updated);
  };

  return (
    <>
      <Button 
        variant={buttonVariant} 
        onClick={handleModalOpen}
        disabled={disabled}
      >
        {buttonText}
      </Button>

      <Modal show={showModal} onHide={handleModalClose} size="lg">
        <Modal.Header closeButton>
          <Modal.Title>Select Models for Explanation</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p className="text-muted">Select models from different providers. You can choose multiple models.</p>
          {currentModels.length > 0 && (
            <div className="mb-3 p-3 bg-light rounded">
              <strong>Current Models:</strong>
              <div className="mt-2">
                {currentModels.map((m, idx) => {
                  const modelName = m.model.replace(/_/g, '.'); // Convert underscores back to dots for display
                  return (
                    <Badge key={idx} bg="info" className="me-2 mb-1">
                      {m.provider} / {modelName}
                    </Badge>
                  );
                })}
              </div>
            </div>
          )}
          <Accordion defaultActiveKey="0">
            <Accordion.Item eventKey="0">
              <Accordion.Header>
                OpenAI ({selectedModels.filter(m => m.startsWith('openai:')).length} selected)
              </Accordion.Header>
              <Accordion.Body>
                <div className="row">
                  {openAIModels.map((model, index) => {
                    const modelKey = `openai:${model.name}`;
                    return (
                      <div className="col-md-6 mb-2" key={modelKey}>
                        <Form.Check
                          type="checkbox"
                          id={`openai-${index}`}
                          label={model.name}
                          checked={selectedModels.includes(modelKey)}
                          onChange={(e) => handleModelToggle(modelKey, e.target.checked)}
                        />
                      </div>
                    );
                  })}
                </div>
              </Accordion.Body>
            </Accordion.Item>
            <Accordion.Item eventKey="1">
              <Accordion.Header>
                Groq ({selectedModels.filter(m => m.startsWith('groq:')).length} selected)
              </Accordion.Header>
              <Accordion.Body>
                <div className="row">
                  {groqModels.map((model, index) => {
                    const modelKey = `groq:${model.name}`;
                    return (
                      <div className="col-md-6 mb-2" key={modelKey}>
                        <Form.Check
                          type="checkbox"
                          id={`groq-${index}`}
                          label={model.name}
                          checked={selectedModels.includes(modelKey)}
                          onChange={(e) => handleModelToggle(modelKey, e.target.checked)}
                        />
                      </div>
                    );
                  })}
                </div>
              </Accordion.Body>
            </Accordion.Item>
            <Accordion.Item eventKey="2">
              <Accordion.Header>
                OpenRouter ({selectedModels.filter(m => m.startsWith('openrouter:')).length} selected)
              </Accordion.Header>
              <Accordion.Body>
                <div className="row">
                  {openrouterModels.map((model, index) => {
                    const modelKey = `openrouter:${model.name}`;
                    return (
                      <div className="col-md-6 mb-2" key={modelKey}>
                        <Form.Check
                          type="checkbox"
                          id={`openrouter-${index}`}
                          label={model.name}
                          checked={selectedModels.includes(modelKey)}
                          onChange={(e) => handleModelToggle(modelKey, e.target.checked)}
                        />
                      </div>
                    );
                  })}
                </div>
              </Accordion.Body>
            </Accordion.Item>
            <Accordion.Item eventKey="3">
              <Accordion.Header>
                Gemini ({selectedModels.filter(m => m.startsWith('gemini:')).length} selected)
              </Accordion.Header>
              <Accordion.Body>
                <div className="row">
                  {geminiModels.map((model, index) => {
                    const modelKey = `gemini:${model.name}`;
                    return (
                      <div className="col-md-6 mb-2" key={modelKey}>
                        <Form.Check
                          type="checkbox"
                          id={`gemini-${index}`}
                          label={model.name}
                          checked={selectedModels.includes(modelKey)}
                          onChange={(e) => handleModelToggle(modelKey, e.target.checked)}
                        />
                      </div>
                    );
                  })}
                </div>
              </Accordion.Body>
            </Accordion.Item>
            <Accordion.Item eventKey="4">
              <Accordion.Header>
                Ollama ({selectedModels.filter(m => m.startsWith('ollama:')).length} selected)
              </Accordion.Header>
              <Accordion.Body>
                <div className="row">
                  {ollamaModels.map((model, index) => {
                    const modelKey = `ollama:${model.name}`;
                    return (
                      <div className="col-md-6 mb-2" key={modelKey}>
                        <Form.Check
                          type="checkbox"
                          id={`ollama-${index}`}
                          label={model.name}
                          checked={selectedModels.includes(modelKey)}
                          onChange={(e) => handleModelToggle(modelKey, e.target.checked)}
                        />
                      </div>
                    );
                  })}
                </div>
              </Accordion.Body>
            </Accordion.Item>
          </Accordion>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={handleModalClose} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button 
            variant="primary" 
            onClick={handleSubmit} 
            disabled={selectedModels.length === 0 || isSubmitting}
          >
            {isSubmitting ? 'Submitting...' : 'Submit'}
          </Button>
        </Modal.Footer>
      </Modal>
    </>
  );
};

export default LLMSelector;
export type { LLMModel };
