import { TestBed } from '@angular/core/testing';

import { AnswerKeyService } from './answer-key.service';

describe('AnswerKeyService', () => {
  let service: AnswerKeyService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(AnswerKeyService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
