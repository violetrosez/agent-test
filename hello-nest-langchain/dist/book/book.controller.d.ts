import { BookService } from './book.service';
import { CreateBookDto } from './dto/create-book.dto';
import { UpdateBookDto } from './dto/update-book.dto';
export declare class BookController {
    private readonly bookService;
    constructor(bookService: BookService);
    private readonly bookRepository;
    create(createBookDto: CreateBookDto): string;
    findAll(): any;
    findOne(id: string): string;
    update(id: string, updateBookDto: UpdateBookDto): string;
    remove(id: string): string;
}
