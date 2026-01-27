from pydantic import BaseModel
from typing import Sequence 
from pytoy_llm.materials.core import TextSectionData, ModelSectionData

class BaseModelBundle[T: BaseModel](BaseModel, frozen=True):
    """Container holding multiple `pydantic.BaseModel` """
    data: Sequence[T]

    @property
    def bundle_kind(self):
        return "BaseModelBundle"
    
    @property
    def text_section_data(self) -> TextSectionData:
        bundle_kind = self.bundle_kind
        structured_text =  self.structured_text
        description = self.description
        return TextSectionData(bundle_kind=bundle_kind,
                                         structured_text=structured_text,
                                         description=description)

    @property
    def model_section_data(self) -> ModelSectionData:
        # Note: `TextFileBundleData` requires a memory space of 
        # text data. 
        # If we would like to use the big data, 
        # `chunk` or `iter` iteration is necessary regarding `data`.
        return ModelSectionData(bundle_kind=self.bundle_kind,
                                description=self.description,
                                data=self.data)
        
    @property
    def description(self) -> str:
        description = ("This section contains multiple instances of `pydantic.BaseModel`\n"
                       "Both of Json Schemas and Json Data are given as below."
                        )
        return description

    @property
    def structured_text(self) -> str:
        """Returns a structured text representation of the documents for LLM consumption."""
        cls_names = [str(type(elem)) for elem in self.data]
        return (
            f"===BaseModelList===\n"
            f"Classes:{','.join(cls_names)}\n"
            f"When you would like to use json-schemas or json-instances, please notify the caller.\n"
            f"Because in this mode, JsonSchema and JsonInstances are ommited."
        )

"""
Mermaid Class Diagram of pytoy_llm compositional structure

```mermaid
classDiagram
    %% --- Core SectionData ---
    class SectionData
    SectionData <|-- TextSectionData
    SectionData <|-- ModelSectionData

    %% --- Section Usage ---
    class SectionUsage

    %% --- BaseModel bundle ---
    class BaseModelBundle
    BaseModelBundle --> SectionData : provides ModelSectionData
    BaseModelBundle --> SectionData : provides TextSectionData

    %% --- Task ---
    class LLMTask

    %% --- Composers ---
    class SectionDataComposer
    SectionDataComposer --> SectionData : compose()
    SectionDataComposer --> SectionUsage : compose_sections_with_usage()

    class TaskPromptComposer
    TaskPromptComposer --> LLMTask : uses
    TaskPromptComposer --> SectionUsage : uses
    TaskPromptComposer --> SectionData : uses directly (via compose_sections_with_usage)

    %% --- Messages ---
    class InputMessage

    %% --- Relationships summary ---
    TaskPromptComposer --> InputMessage : compose_messages()
"""
