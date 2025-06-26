import dataset.cleaning as cleaning


class Dataset:
    """Dataset
    
    Abstract class for datasets. Subclasses should implement `load` function that load `id_to_fact_check`, `id_to_post`,
    and `fact_check_post_mapping` object attributes. The class also implemenets basic cleaning methods that might be
    reused.
    
    Attributes:
        clean_ocr: bool = True  Should cleaning of OCRs be performed
        remove_emojis: bool = False  Should emojis be removed from the texts?
        remove_urls: bool = True  Should URLs be removed from the texts?
        replace_whitespaces: bool = True  Should whitespaces be replaced by a single space whitespace?
        clean_twitter: bool = True  
        remove_elongation: bool = False  Should occurrence of a string of consecutive identical non-space 
    characters (at least three in a row) with just one instance of that character?

        After `load` is called, following attributes are accesible:
                fact_check_post_mapping: list[tuple[int, int]]  List of Factcheck-Post id pairs.
                id_to_fact_check: dict[int, str]  Factcheck id -> Factcheck text
                id_to_post: dict[int, str]  Post id -> Post text
                
    Methods:
        clean_text: Performs text cleaning based on initialization attributes.
        maybe_clean_ocr: Perform OCR-specific text cleaning, if `self.clean_ocr`
        load: Abstract method. To be implemented by the subclasses.
        
    """
    
    # The default values here are based on our preliminary experiments. Might not be the best for all cases.
    def __init__(
        self,
        clean_ocr: bool = False,
        dataset: str = None,  # Here to read and discard the `dataset` field from the argparser
        remove_emojis: bool = True,
        remove_urls: bool = False,
        replace_whitespaces: bool = True,
        clean_twitter: bool = True,
        remove_elongation: bool = False
    ):
        self.clean_ocr = clean_ocr
        self.remove_emojis = remove_emojis
        self.remove_urls = remove_urls
        self.replace_whitespaces = replace_whitespaces
        self.clean_twitter = clean_twitter
        self.remove_elongation = remove_elongation
        
        
    def __len__(self):
        return len(self.fact_check_post_mapping)

    
    def getitem_with_ids(self, idx):
        fc_id, p_id = self.fact_check_post_mapping[idx]
        return self.id_to_fact_check_vocab[fc_id], self.id_to_post_vocab[p_id],self.fact_check_post_mapping[idx]
    
    def __getitem__(self, idx):
        fc_id, p_id = self.fact_check_post_mapping[idx]
        return self.id_to_fact_check_vocab[fc_id], self.id_to_post_vocab[p_id]



    def clean_text(self, text):
        
        if self.remove_urls:
            text = cleaning.remove_urls(text)

        if self.remove_emojis:
            text = cleaning.remove_emojis(text)

        if self.replace_whitespaces:
            text = cleaning.replace_whitespaces(text)
        
        if self.clean_twitter:
            text = cleaning.clean_twitter_picture_links(text)
            text = cleaning.clean_twitter_links(text)
        
        if self.remove_elongation:
            text = cleaning.remove_elongation(text)

        return text.strip()
            

    def maybe_clean_ocr(self, ocr):
        if self.clean_ocr:
            return cleaning.clean_ocr(ocr)
        return ocr
        
    
    def __getattr__(self, name):
        if name in {'id_to_fact_check_vocab', 'id_to_post_vocab', 'fact_check_post_mapping'}:
            raise AttributeError(f"You have to `load` the dataset first before using '{name}'")
        # raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        # print('lol')

        
    def load(self):
        raise NotImplementedError
        
