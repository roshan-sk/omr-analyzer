import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AnswerKey } from './answer-key';

describe('AnswerKey', () => {
  let component: AnswerKey;
  let fixture: ComponentFixture<AnswerKey>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AnswerKey],
    }).compileComponents();

    fixture = TestBed.createComponent(AnswerKey);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
