from pydantic import BaseModel, ConfigDict, Field

LocalizedText = dict[str, str]


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class VideoUrl(_Base):
    content_type: str = Field(default="", alias="contentType")
    content: LocalizedText = Field(default_factory=dict)


class VideoTranscript(_Base):
    content_id: str = Field(default="", alias="contentId")
    video_id: str = Field(default="", alias="videoId")
    duration: int = 0
    language_code: str = Field(default="", alias="languageCode")
    title: LocalizedText = Field(default_factory=dict)
    transcript_text: str = Field(default="", alias="transcriptText")
    video_url: list[VideoUrl] = Field(default_factory=list, alias="videoUrl")
    updated_at: str = Field(default="", alias="updatedAt")


class NoteText(_Base):
    content_type: str = Field(default="", alias="contentType")
    content: LocalizedText = Field(default_factory=dict)


class RevisionNote(_Base):
    content_id: str = Field(default="", alias="contentId")
    language_code: str = Field(default="", alias="languageCode")
    updated_at: str = Field(default="", alias="updatedAt")
    title: LocalizedText = Field(default_factory=dict)
    note_text: list[NoteText] = Field(default_factory=list, alias="noteText")


class Question(_Base):
    question_id: str = Field(default="", alias="questionId")
    question_type: str = Field(default="", alias="questionType")
    question_text: LocalizedText = Field(default_factory=dict, alias="questionText")
    solution_description: LocalizedText = Field(default_factory=dict, alias="solutionDescription")
    solution_index: list[int] = Field(default_factory=list, alias="solutionIndex")
    difficulty_level: int = Field(default=0, alias="difficultyLevel")
    language_code: list[str] = Field(default_factory=list, alias="languageCode")
    updated_at: str = Field(default="", alias="updatedAt")


class SubConceptContent(_Base):
    revision_notes: list[RevisionNote] = Field(default_factory=list, alias="revisionNotes")
    video_transcripts: list[VideoTranscript] = Field(default_factory=list, alias="videoTranscripts")
    questions: list[Question] = Field(default_factory=list)
    images: list[dict] = Field(default_factory=list)
    pdfs: list[dict] = Field(default_factory=list)
    solutions: list[dict] = Field(default_factory=list)
    tables: list[dict] = Field(default_factory=list)


class SubConcept(_Base):
    sub_concept_id: str = Field(default="", alias="subConceptId")
    sub_concept_name: str = Field(default="", alias="subConceptName")
    sub_concept_sequence: int = Field(default=0, alias="subConceptSequence")
    content: SubConceptContent = Field(default_factory=SubConceptContent)


class Concept(_Base):
    concept_id: str = Field(default="", alias="conceptId")
    concept_name: str = Field(default="", alias="conceptName")
    concept_sequence: int = Field(default=0, alias="conceptSequence")
    ebook_pdf: str = Field(default="", alias="ebookPdf")
    sub_concepts: list[SubConcept] = Field(default_factory=list, alias="subConcepts")


class Chapter(_Base):
    chapter_id: str = Field(default="", alias="chapterId")
    chapter_name: str = Field(default="", alias="chapterName")
    chapter_sequence: int = Field(default=0, alias="chapterSequence")
    ebook_pdf: str = Field(default="", alias="ebookPdf")
    concepts: list[Concept] = Field(default_factory=list)


class ContentTypePriority(_Base):
    content_type: str = Field(default="", alias="contentType")
    priority: int = 0
    weight: float = 0.0


class RetrievalConfiguration(_Base):
    search_mode: str = Field(default="", alias="searchMode")
    top_k: int = Field(default=0, alias="topK")
    minimum_similarity_threshold: float = Field(default=0.0, alias="minimumSimilarityThreshold")
    deduplicate_results: bool = Field(default=True, alias="deduplicateResults")
    enable_source_diversification: bool = Field(default=True, alias="enableSourceDiversification")
    allow_cross_subject_search: bool = Field(default=False, alias="allowCrossSubjectSearch")
    content_priority_order: list[ContentTypePriority] = Field(
        default_factory=list, alias="contentPriorityOrder"
    )


class ContentPackage(_Base):
    cp_id: str = Field(default="", alias="cp_id")
    cp_name: str = Field(default="", alias="cpName")
    plan_id: str = Field(default="", alias="planId")
    plan_name: str = Field(default="", alias="planName")
    board_id: str = Field(default="", alias="boardId")
    board_name: str = Field(default="", alias="boardName")
    grade_id: str = Field(default="", alias="gradeId")
    grade_name: str = Field(default="", alias="gradeName")
    medium_id: str = Field(default="", alias="mediumId")
    medium_name: str = Field(default="", alias="mediumName")
    subject_id: str = Field(default="", alias="subjectId")
    subject_name: str = Field(default="", alias="subjectName")
    stream_id: str = Field(default="", alias="streamId")
    stream_name: str = Field(default="", alias="streamName")
    academic_year: str = Field(default="", alias="academicYear")
    curriculum_version: str = Field(default="", alias="curriculumVersion")
    language_code: str = Field(default="", alias="languageCode")
    business_account_id: str = Field(default="", alias="businessAccountId")
    is_active: bool = Field(default=False, alias="isActive")
    retrieval_configuration: RetrievalConfiguration = Field(
        default_factory=RetrievalConfiguration, alias="retrievalConfiguration"
    )
    chapters: list[Chapter] = Field(default_factory=list)


class BundleResponse(_Base):
    status_code: int = Field(default=0, alias="statusCode")
    is_success: bool = Field(default=False, alias="isSuccess")
    message: str = ""
    result: list[ContentPackage] = Field(default_factory=list)
